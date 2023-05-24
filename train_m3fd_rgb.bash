export CUDA_VISIBLE_DEVICES=0
python train.py root='' --dataset=m3fd_rgb --model=efficientdetv2_dt --batch-size=16 --amp --lr=1e-4 --opt adam --sched plateau --num-classes=6 --mean 0.49151019 0.50717567 0.50293698 --std 0.1623529 0.14178433 0.13799928 --save-images --workers=4 --pretrained
