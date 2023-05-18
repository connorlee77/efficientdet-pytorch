export CUDA_VISIBLE_DEVICES=0
python validate.py root='' --dataset=flir_aligned --model efficientdetv2_dt --workers 4 --checkpoint output/train/20221123-140059-efficientdetv2_dt/model_best.pth.tar --num-classes=90 -b 16 --results stuff
