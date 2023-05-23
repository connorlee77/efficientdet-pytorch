export CUDA_VISIBLE_DEVICES=1
python validate.py root='' --dataset=seeingthroughfog_gated_easy --model tf_efficientdet_d5 --workers 4 --checkpoint output/train/20230521-213630-tf_efficientdet_d5/model_best.pth.tar --num-classes=4 -b 4 --results stuff --split test --img-size 1280

python validate.py root='' --dataset=seeingthroughfog_gated_moderate --model tf_efficientdet_d5 --workers 4 --checkpoint output/train/20230521-213630-tf_efficientdet_d5/model_best.pth.tar --num-classes=4 -b 4 --results stuff --split test --img-size 1280

python validate.py root='' --dataset=seeingthroughfog_gated_hard --model tf_efficientdet_d5 --workers 4 --checkpoint output/train/20230521-213630-tf_efficientdet_d5/model_best.pth.tar --num-classes=4 -b 4 --results stuff --split test --img-size 1280
