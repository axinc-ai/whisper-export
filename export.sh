#for i in large large-v3
for i in tiny base small medium
do
python3 cli.py audio.wav --model $i --export_encoder
python3 cli.py audio.wav --model $i --export_decoder
done
