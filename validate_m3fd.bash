export CUDA_VISIBLE_DEVICES=1
python validate_cls.py root='' --dataset=m3fd_rgb --model efficientdetv2_dt --workers 8 --checkpoint /home/ganlu/workspace/efficientdet-pytorch/output/train/m3fd-rgb-backbone-cls/model_best.pth.tar --num-classes=6 -b 8 --results m3fd_rgb_test.txt --split test
