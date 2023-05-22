export CUDA_VISIBLE_DEVICES=1
python validate.py root='' --dataset=seeingthroughfog_rgb_easy --model efficientdetv2_dt --workers 4 --checkpoint output/train/rgb/model_best.pth.tar --num-classes=4 -b 16 --results stuff --split test --img-size 1280

python validate.py root='' --dataset=seeingthroughfog_rgb_moderate --model efficientdetv2_dt --workers 4 --checkpoint output/train/rgb/model_best.pth.tar --num-classes=4 -b 16 --results stuff --split test --img-size 1280

python validate.py root='' --dataset=seeingthroughfog_rgb_hard --model efficientdetv2_dt --workers 4 --checkpoint output/train/rgb/model_best.pth.tar --num-classes=4 -b 16 --results stuff --split test --img-size 1280
