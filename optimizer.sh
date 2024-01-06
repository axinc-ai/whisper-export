mkdir optimize_model
#for i in large large-v3
for i in tiny base small medium
do
python3 onnx_optimizer.py export_model/encoder_${i}_opset17.onnx
python3 onnx_optimizer.py -m optimizer/manual_opt_${i}.json export_model/decoder_${i}_opset17.onnx
mv export_model/encoder_${i}_opset17.opt.onnx optimize_model/encoder_${i}.opt3.onnx
mv export_model/decoder_${i}_opset17.opt.onnx optimize_model/decoder_${i}_fix_kv_cache.opt3.onnx
python3 onnx2prototxt.py optimize_model/encoder_${i}.opt3.onnx
python3 onnx2prototxt.py optimize_model/decoder_${i}_fix_kv_cache.opt3.onnx
done
