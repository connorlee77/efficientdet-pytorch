# export CUDA_VISIBLE_DEVICES=0
# python train.py root='' --eval-metric=acc --bench-task=train_cls --dataset=m3fd_rgb --model=efficientdetv2_dt \
# --batch-size=16 --amp --lr=1e-4 --opt adam --sched plateau --num-classes=6 \
# --mean 0.49151019 0.50717567 0.50293698 --std 0.1623529 0.14178433 0.13799928 \
# --save-images --workers=4 --initial-checkpoint /home/ganlu/workspace/efficientdet-pytorch/output/train/20230524-003411-efficientdetv2_dt/model_best.pth.tar


export CUDA_VISIBLE_DEVICES=0
python train.py root='' --eval-metric=acc --bench-task=train_cls --dataset=seeingthroughfog_rgb_all --model=efficientdetv2_dt \
--batch-size=12 --amp --lr=1e-4 --opt adam --sched plateau --num-classes=4 --num-scenes=7 \
--save-images --workers=12 --image-size 1280 \
--initial-checkpoint /home/ganlu/workspace/efficientdet-pytorch/output/train/stf_1280_all_scenes_weights/rgb/model_best.pth.tar