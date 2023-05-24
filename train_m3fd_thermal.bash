export CUDA_VISIBLE_DEVICES=0
python train.py root='' --dataset=m3fd_thermal --model=efficientdetv2_dt --batch-size=16 --amp --lr=1e-4 --opt adam --sched plateau --num-classes=6 --mean 0.33000296 0.33000296 0.33000296 --std 0.18958051 0.18958051 0.18958051 --save-images --workers=4 --pretrained
