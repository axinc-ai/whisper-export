import argparse
import os
import warnings
from typing import List, Optional, Tuple, Union, TYPE_CHECKING

import numpy as np
import torch
import tqdm

from .audio import (
    FRAMES_PER_SECOND,
    HOP_LENGTH,
    N_FRAMES,
    N_SAMPLES,
    SAMPLE_RATE,
    log_mel_spectrogram,
    pad_or_trim,
)
from . import decoding
from . import model as mod_model
from .decoding import DecodingOptions, DecodingResult
from .tokenizer import LANGUAGES, TO_LANGUAGE_CODE, get_tokenizer
from .utils import exact_div, format_timestamp, optional_int, optional_float, str2bool, write_txt, write_vtt, write_srt

if TYPE_CHECKING:
    from .model import Whisper


def transcribe(
    model: "Whisper",
    audio: Union[str, np.ndarray, torch.Tensor],
    *,
    verbose: Optional[bool] = None,
    temperature: Union[float, Tuple[float, ...]] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
    compression_ratio_threshold: Optional[float] = 2.4,
    logprob_threshold: Optional[float] = -1.0,
    no_speech_threshold: Optional[float] = 0.6,
    condition_on_previous_text: bool = True,
    **decode_options,
):
    """
    Transcribe an audio file using Whisper

    Parameters
    ----------
    model: Whisper
        The Whisper model instance

    audio: Union[str, np.ndarray, torch.Tensor]
        The path to the audio file to open, or the audio waveform

    verbose: bool
        Whether to display the text being decoded to the console. If True, displays all the details,
        If False, displays minimal details. If None, does not display anything

    temperature: Union[float, Tuple[float, ...]]
        Temperature for sampling. It can be a tuple of temperatures, which will be successfully used
        upon failures according to either `compression_ratio_threshold` or `logprob_threshold`.

    compression_ratio_threshold: float
        If the gzip compression ratio is above this value, treat as failed

    logprob_threshold: float
        If the average log probability over sampled tokens is below this value, treat as failed

    no_speech_threshold: float
        If the no_speech probability is higher than this value AND the average log probability
        over sampled tokens is below `logprob_threshold`, consider the segment as silent

    condition_on_previous_text: bool
        if True, the previous output of the model is provided as a prompt for the next window;
        disabling may make the text inconsistent across windows, but the model becomes less prone to
        getting stuck in a failure loop, such as repetition looping or timestamps going out of sync.

    decode_options: dict
        Keyword arguments to construct `DecodingOptions` instances

    Returns
    -------
    A dictionary containing the resulting text ("text") and segment-level details ("segments"), and
    the spoken language ("language"), which is detected when `decode_options["language"]` is None.
    """
    dtype = torch.float16 if decode_options.get("fp16", True) else torch.float32
    if model.device == torch.device("cpu"):
        if torch.cuda.is_available():
            warnings.warn("Performing inference on CPU when CUDA is available")
        if dtype == torch.float16:
            warnings.warn("FP16 is not supported on CPU; using FP32 instead")
            dtype = torch.float32

    if dtype == torch.float32:
        decode_options["fp16"] = False

    mel = log_mel_spectrogram(audio, model.dims.n_mels, padding=N_SAMPLES)
    content_frames = mel.shape[-1] - N_FRAMES

    if decode_options.get("language", None) is None:
        if verbose:
            print("Detecting language using up to the first 30 seconds. Use `--language` to specify the language")
        segment = pad_or_trim(mel, N_FRAMES).to(model.device).to(dtype)
        _, probs = model.detect_language(segment)
        decode_options["language"] = max(probs, key=probs.get)
        if verbose is not None:
            print(f"Detected language: {LANGUAGES[decode_options['language']].title()}")

    mel = mel.unsqueeze(0)
    language = decode_options["language"]
    task = decode_options.get("task", "transcribe")
    tokenizer = get_tokenizer(
        model.is_multilingual,
        num_languages=model.num_languages,
        language=language,
        task=task,
    )

    #def decode_with_fallback(segment: torch.Tensor) -> List[DecodingResult]:
    #    temperatures = [temperature] if isinstance(temperature, (int, float)) else temperature
    #    kwargs = {**decode_options}
    #    t = temperatures[0]
    #    if t == 0:
    #        best_of = kwargs.pop("best_of", None)
    #    else:
    #        best_of = kwargs.get("best_of", None)

    #    options = DecodingOptions(**kwargs, temperature=t)
    #    results = model.decode(segment, options)

    #    kwargs.pop("beam_size", None)  # no beam search for t > 0
    #    kwargs.pop("patience", None)  # no patience for t > 0
    #    kwargs["best_of"] = best_of  # enable best_of for t > 0
    #    for t in temperatures[1:]:
    #        needs_fallback = [
    #            compression_ratio_threshold is not None
    #            and result.compression_ratio > compression_ratio_threshold
    #            or logprob_threshold is not None
    #            and result.avg_logprob < logprob_threshold
    #            for result in results
    #        ]
    #        if any(needs_fallback):
    #            print("temperature", t)
    #            options = DecodingOptions(**kwargs, temperature=t)
    #            retries = model.decode(segment[needs_fallback], options)
    #            for retry_index, original_index in enumerate(np.nonzero(needs_fallback)[0]):
    #                results[original_index] = retries[retry_index]

    #    return results

    def decode_with_fallback(segment: torch.Tensor) -> List[DecodingResult]:
        temperatures = (
            [temperature] if isinstance(temperature, (int, float)) else temperature
        )
        decode_result = None

        for t in temperatures:
            kwargs = {**decode_options}
            if t > 0:
                # disable beam_size and patience when t > 0
                kwargs.pop("beam_size", None)
                kwargs.pop("patience", None)
                print("temperature", t)
            else:
                # disable best_of when t == 0
                kwargs.pop("best_of", None)

            options = DecodingOptions(**kwargs, temperature=t)
            decode_result = model.decode(segment, options)[0]

            needs_fallback = False
            if (
                compression_ratio_threshold is not None
                and decode_result.compression_ratio > compression_ratio_threshold
            ):
                needs_fallback = True  # too repetitive
            if (
                logprob_threshold is not None
                and decode_result.avg_logprob < logprob_threshold
            ):
                needs_fallback = True  # average log probability is too low
            if (
                no_speech_threshold is not None
                and decode_result.no_speech_prob > no_speech_threshold
            ):
                needs_fallback = False  # silence
            if not needs_fallback:
                break

        return [decode_result]

    seek = 0
    input_stride = exact_div(
        N_FRAMES, model.dims.n_audio_ctx
    )  # mel frames per output token: 2
    time_precision = (
        input_stride * HOP_LENGTH / SAMPLE_RATE
    )  # time per output token: 0.02 (seconds)
    all_tokens = []
    all_segments = []
    prompt_reset_since = 0

    initial_prompt = decode_options.pop("initial_prompt", None) or []
    if initial_prompt:
        initial_prompt = tokenizer.encode(" " + initial_prompt.strip())
        all_tokens.extend(initial_prompt)

    def new_segment(
        *, start: float, end: float, tokens: torch.Tensor, result: DecodingResult
    ):
        tokens = tokens.tolist()
        text_tokens = [token for token in tokens if token < tokenizer.eot]
        return {
            "seek": seek,
            "start": start,
            "end": end,
            "text": tokenizer.decode(text_tokens),
            "tokens": tokens,
            "temperature": result.temperature,
            "avg_logprob": result.avg_logprob,
            "compression_ratio": result.compression_ratio,
            "no_speech_prob": result.no_speech_prob,
        }

    # show the progress bar when verbose is False (if True, transcribed text will be printed)
    with tqdm.tqdm(
        total=content_frames, unit="frames", disable=verbose is not False
    ) as pbar:
        last_speech_timestamp = 0.0
        while seek < content_frames:
            time_offset = float(seek * HOP_LENGTH / SAMPLE_RATE)
            mel_segment = mel[:, :, seek : seek + N_FRAMES]
            segment_size = min(N_FRAMES, content_frames - seek)
            segment_duration = segment_size * HOP_LENGTH / SAMPLE_RATE
            mel_segment = pad_or_trim(mel_segment, N_FRAMES).to(model.device).to(dtype)

            decode_options["prompt"] = all_tokens[prompt_reset_since:]
            result: DecodingResult = decode_with_fallback(mel_segment)[0]
            tokens = torch.tensor(result.tokens)

            if no_speech_threshold is not None:
                # no voice activity check
                should_skip = result.no_speech_prob > no_speech_threshold
                if (
                    logprob_threshold is not None
                    and result.avg_logprob > logprob_threshold
                ):
                    # don't skip if the logprob is high enough, despite the no_speech_prob
                    should_skip = False

                if should_skip:
                    seek += segment_size  # fast-forward to the next segment boundary
                    continue

            previous_seek = seek
            current_segments = []

            timestamp_tokens: torch.Tensor = tokens.ge(tokenizer.timestamp_begin)
            single_timestamp_ending = timestamp_tokens[-2:].tolist() == [False, True]

            consecutive = torch.where(timestamp_tokens[:-1] & timestamp_tokens[1:])[0]
            consecutive.add_(1)
            if len(consecutive) > 0:
                # if the output contains two consecutive timestamp tokens
                slices = consecutive.tolist()
                if single_timestamp_ending:
                    slices.append(len(tokens))

                last_slice = 0
                for current_slice in slices:
                    sliced_tokens = tokens[last_slice:current_slice]
                    start_timestamp_pos = (
                        sliced_tokens[0].item() - tokenizer.timestamp_begin
                    )
                    end_timestamp_pos = (
                        sliced_tokens[-1].item() - tokenizer.timestamp_begin
                    )
                    current_segments.append(
                        new_segment(
                            start=time_offset + start_timestamp_pos * time_precision,
                            end=time_offset + end_timestamp_pos * time_precision,
                            tokens=sliced_tokens,
                            result=result,
                        )
                    )
                    last_slice = current_slice

                if single_timestamp_ending:
                    # single timestamp at the end means no speech after the last timestamp.
                    seek += segment_size
                else:
                    # otherwise, ignore the unfinished segment and seek to the last timestamp
                    last_timestamp_pos = (
                        tokens[last_slice - 1].item() - tokenizer.timestamp_begin
                    )
                    seek += last_timestamp_pos * input_stride
            else:
                duration = segment_duration
                timestamps = tokens[timestamp_tokens.nonzero().flatten()]
                if (
                    len(timestamps) > 0
                    and timestamps[-1].item() != tokenizer.timestamp_begin
                ):
                    # no consecutive timestamps but it has a timestamp; use the last one.
                    last_timestamp_pos = (
                        timestamps[-1].item() - tokenizer.timestamp_begin
                    )
                    duration = last_timestamp_pos * time_precision

                current_segments.append(
                    new_segment(
                        start=time_offset,
                        end=time_offset + duration,
                        tokens=tokens,
                        result=result,
                    )
                )
                seek += segment_size

            if verbose:
                for segment in current_segments:
                    start, end, text = segment["start"], segment["end"], segment["text"]
                    line = f"[{format_timestamp(start)} --> {format_timestamp(end)}] {text}"
                    print(line)

            # if a segment is instantaneous or does not contain text, clear it
            for i, segment in enumerate(current_segments):
                if segment["start"] == segment["end"] or segment["text"].strip() == "":
                    segment["text"] = ""
                    segment["tokens"] = []
                    segment["words"] = []

            all_segments.extend(
                [
                    {"id": i, **segment}
                    for i, segment in enumerate(
                        current_segments, start=len(all_segments)
                    )
                ]
            )
            all_tokens.extend(
                [token for segment in current_segments for token in segment["tokens"]]
            )

            if not condition_on_previous_text or result.temperature > 0.5:
                # do not feed the prompt tokens if a high temperature was used
                prompt_reset_since = len(all_tokens)

            # update progress bar
            pbar.update(min(content_frames, seek) - previous_seek)

    return dict(
        text=tokenizer.decode(all_tokens[len(initial_prompt) :]),
        segments=all_segments,
        language=language,
    )


def cli():
    from . import available_models

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("audio", nargs="+", type=str, help="audio file(s) to transcribe")
    parser.add_argument("--model", default="small", choices=available_models(), help="name of the Whisper model to use")
    parser.add_argument("--opset", default=17, choices=[11, 17], help="opset of the ONNX")
    parser.add_argument("--device", default="cpu", help="device to use for PyTorch inference")
    parser.add_argument("--output_dir", "-o", type=str, default=".", help="directory to save the outputs")
    parser.add_argument("--verbose", type=str2bool, default=True, help="whether to print out the progress and debug messages")

    parser.add_argument("--task", type=str, default="transcribe", choices=["transcribe", "translate"], help="whether to perform X->X speech recognition ('transcribe') or X->English translation ('translate')")
    parser.add_argument("--language", type=str, default=None, choices=sorted(LANGUAGES.keys()) + sorted([k.title() for k in TO_LANGUAGE_CODE.keys()]), help="language spoken in the audio, specify None to perform language detection")

    parser.add_argument("--temperature", type=float, default=0, help="temperature to use for sampling")
    parser.add_argument("--best_of", type=optional_int, default=5, help="number of candidates when sampling with non-zero temperature")
    parser.add_argument("--beam_size", type=optional_int, default=5, help="number of beams in beam search, only applicable when temperature is zero")
    parser.add_argument("--patience", type=float, default=None, help="optional patience value to use in beam decoding, as in https://arxiv.org/abs/2204.05424, the default (1.0) is equivalent to conventional beam search")
    parser.add_argument("--length_penalty", type=float, default=None, help="optional token length penalty coefficient (alpha) as in https://arxiv.org/abs/1609.08144, uses simple lengt normalization by default")

    parser.add_argument("--suppress_tokens", type=str, default="-1", help="comma-separated list of token ids to suppress during sampling; '-1' will suppress most special characters except common punctuations")
    parser.add_argument("--initial_prompt", type=str, default=None, help="optional text to provide as a prompt for the first window.")
    parser.add_argument("--condition_on_previous_text", type=str2bool, default=True, help="if True, provide the previous output of the model as a prompt for the next window; disabling may make the text inconsistent across windows, but the model becomes less prone to getting stuck in a failure loop")
    parser.add_argument("--fp16", type=str2bool, default=True, help="whether to perform inference in fp16; True by default")

    parser.add_argument("--temperature_increment_on_fallback", type=optional_float, default=0.2, help="temperature to increase when falling back when the decoding fails to meet either of the thresholds below")
    parser.add_argument("--compression_ratio_threshold", type=optional_float, default=2.4, help="if the gzip compression ratio is higher than this value, treat the decoding as failed")
    parser.add_argument("--logprob_threshold", type=optional_float, default=-1.0, help="if the average log probability is lower than this value, treat the decoding as failed")
    parser.add_argument("--no_speech_threshold", type=optional_float, default=0.6, help="if the probability of the <|nospeech|> token is higher than this value AND the decoding has failed due to `logprob_threshold`, consider the segment as silence")

    parser.add_argument("--export_encoder",  action='store_true')
    parser.add_argument("--export_decoder",  action='store_true')

    parser.add_argument("--import_encoder",  action='store_true')
    parser.add_argument("--import_decoder",  action='store_true')

    parser.add_argument("--fine_tuning",  type=str, default=None)
    
    args = parser.parse_args().__dict__
    model_name: str = args.pop("model")
    output_dir: str = args.pop("output_dir")
    device: str = args.pop("device")
    os.makedirs(output_dir, exist_ok=True)

    mod_model.model_name = model_name

    opset: int = args.pop("opset")

    decoding.model_name = model_name
    decoding.opset = opset

    if args.pop("export_encoder"):
        decoding.export_encoder = True
    if args.pop("export_decoder"):
        decoding.export_decoder = True
    if args.pop("import_encoder"):
        decoding.import_encoder = True
    if args.pop("import_decoder"):
        decoding.import_decoder = True

    if model_name.endswith(".en") and args["language"] not in {"en", "English"}:
        warnings.warn(f"{model_name} is an English-only model but receipted '{args['language']}'; using English instead.")
        args["language"] = "en"

    temperature = args.pop("temperature")
    temperature_increment_on_fallback = args.pop("temperature_increment_on_fallback")
    if temperature_increment_on_fallback is not None:
        temperature = tuple(np.arange(temperature, 1.0 + 1e-6, temperature_increment_on_fallback))
    else:
        temperature = [temperature]

    from . import load_model
    model = load_model(model_name, device=device)
    fine_tuning = args.pop("fine_tuning")
    if fine_tuning:
        state_dict = torch.load(fine_tuning, map_location=device)
        missing, unexpected = model.load_state_dict(state_dict, strict = True)
        if len(missing) or len(unexpected):
            print("Weight name not found", missing, unexpected)
            raise

    for audio_path in args.pop("audio"):
        result = transcribe(model, audio_path, temperature=temperature, **args)

        audio_basename = os.path.basename(audio_path)

        # save TXT
        with open(os.path.join(output_dir, audio_basename + ".txt"), "w", encoding="utf-8") as txt:
            write_txt(result["segments"], file=txt)

        # save VTT
        with open(os.path.join(output_dir, audio_basename + ".vtt"), "w", encoding="utf-8") as vtt:
            write_vtt(result["segments"], file=vtt)

        # save SRT
        with open(os.path.join(output_dir, audio_basename + ".srt"), "w", encoding="utf-8") as srt:
            write_srt(result["segments"], file=srt)


if __name__ == '__main__':
    cli()
