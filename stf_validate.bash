export CUDA_VISIBLE_DEVICES=0
# python validate.py root='' --dataset=seeingthroughfog_rgb --model efficientdetv2_dt --workers 4 --checkpoint output/train/20230610-173215-efficientdetv2_dt/model_best.pth.tar --num-classes=4 -b 4 --results rgb_test_clear_easy.txt --split test --img-size 1280

# python validate.py root='' --dataset=seeingthroughfog_gated --model efficientdetv2_dt --workers 16 --checkpoint output/train/20230610-173140-efficientdetv2_dt/model_best.pth.tar --num-classes=4 -b 16 --results rgb_test_clear_easy.txt --split test --img-size 1280



python validate.py root='' --dataset=stf_snow_rain_rgb --model efficientdetv2_dt --workers 16 --checkpoint stf_1280_clear_weights/20230622-162846-efficientdetv2_dt/model_best.pth.tar --num-classes=4 -b 16 --results rgb_test_clear_easy.txt --split test --img-size 1280

python validate.py root='' --dataset=stf_snow_rain_gated --model efficientdetv2_dt --workers 16 --checkpoint stf_1280_clear_weights/20230622-162902-efficientdetv2_dt/model_best.pth.tar --num-classes=4 -b 16 --results rgb_test_clear_easy.txt --split test --img-size 1280
