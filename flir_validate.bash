export CUDA_VISIBLE_DEVICES=1
# python validate.py root='' --dataset=flir_aligned_rgb --model efficientdetv2_dt --workers 16 --checkpoint output/train/flir_trained/flir-rgb/model_best.pth.tar --num-classes=3 -b 16 --results flir.txt --split test --mean 0.62721553 0.63597459 0.62891984 --std 0.16723704 0.17459581 0.18347738

python validate.py root='' --dataset=flir_aligned_thermal --model efficientdetv2_dt --workers 16 --checkpoint output/train/flir_trained/flir-thermal/model_best.pth.tar --num-classes=3 -b 16 --results flir.txt --split test --mean 0.53584253 0.53584253 0.53584253 --std 0.24790472 0.24790472 0.24790472
