# export CUDA_VISIBLE_DEVICES=1
# python validate_cls.py root='' --dataset=m3fd_rgb --model efficientdetv2_dt --workers 8 \
# --checkpoint /home/ganlu/workspace/efficientdet-pytorch/output/train/m3fd-rgb-backbone-cls/model_best.pth.tar --num-classes=6 -b 8 --results m3fd_rgb_test_cls.txt --split test

# export CUDA_VISIBLE_DEVICES=1
# python validate_cls.py root='' --dataset=flir_aligned_rgb --model efficientdetv2_dt --workers 8 \
# --checkpoint /home/ganlu/workspace/efficientdet-pytorch/output/train/flir-rgb-backbone-cls/model_best.pth.tar --num-classes=3 --num-scenes=3 \
# -b 8 --results flir_rgb_test_cls.txt --split test

export CUDA_VISIBLE_DEVICES=1
python validate_cls.py root='' --dataset=stf_clear_rgb --model efficientdetv2_dt --workers 8 \
--checkpoint /home/ganlu/workspace/efficientdet-pytorch/output/train/stf-rgb-backbone-cls/model_best.pth.tar --num-classes=4 --num-scenes=3 \
-b 12 --results stf_rgb_test_cls.txt --split test --img-size 1280
