# Whisper ONNX Export Script

Export whisper to onnx. The decoder fixes the size of kv_cache to avoid re-allocating tensors for each inference.

## Requirements

- Windows or macOS or Linux
- torch 2.0
- onnx 1.13.1

## ONNX Export

This repository based on [whisper.openvino](https://github.com/zhuzilin/whisper-openvino), but
OpenVinoAudioEncoder and OpenVinoTextDecoder were replaced by official AudioEncoder and TextDecoder for ONNX export.

The following command will onnx export:

```
python3 cli.py audio.wav --model medium --export_encoder
python3 cli.py audio.wav --model medium --export_decoder
```

The following command will onnx import for inference test:

```
python3 cli.py audio.wav --model medium --import_encoder
python3 cli.py audio.wav --model medium --import_decoder
```

## ONNX Export Examples

- export.sh : Export to onnx
- verify.sh : Verify onnx output
- optimize.sh : Optimize onnx using ailia onnx optimizer

## Export Fine Tuned Model

You can also read weights saved_state_dicted from the original whisper.

```
python3 cli.py audio.wav --model medium --export_decoder --fine_tuning model.pth
```

# Whisper Original information

[ORIGINAL.md](ORIGINAL.md)
