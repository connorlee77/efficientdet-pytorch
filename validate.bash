export CUDA_VISIBLE_DEVICES=1
python validate.py root='' --dataset=seeingthroughfog_rgb_easy --model tf_efficientdet_d5 --workers 8 --checkpoint output/train/20230521-213509-tf_efficientdet_d5/model_best.pth.tar --num-classes=4 -b 8 --results rgb_test_clear_easy.txt --split test --img-size 1280

python validate.py root='' --dataset=seeingthroughfog_rgb_easy --model tf_efficientdet_d5 --workers 8 --checkpoint output/train/20230521-213509-tf_efficientdet_d5/model_best.pth.tar --num-classes=4 -b 8 --results rgb_light_fog_easy.txt --split lightfog --img-size 1280

python validate.py root='' --dataset=seeingthroughfog_rgb_easy --model tf_efficientdet_d5 --workers 8 --checkpoint output/train/20230521-213509-tf_efficientdet_d5/model_best.pth.tar --num-classes=4 -b 8 --results rgb_dense_fog_easy.txt --split densefog --img-size 1280

python validate.py root='' --dataset=seeingthroughfog_rgb_easy --model tf_efficientdet_d5 --workers 8 --checkpoint output/train/20230521-213509-tf_efficientdet_d5/model_best.pth.tar --num-classes=4 -b 8 --results rgb_snow_rain_easy.txt --split snowrain --img-size 1280



python validate.py root='' --dataset=seeingthroughfog_rgb_moderate --model tf_efficientdet_d5 --workers 8 --checkpoint output/train/20230521-213509-tf_efficientdet_d5/model_best.pth.tar --num-classes=4 -b 8 --results rgb_test_clear_moderate.txt --split test --img-size 1280

python validate.py root='' --dataset=seeingthroughfog_rgb_moderate --model tf_efficientdet_d5 --workers 8 --checkpoint output/train/20230521-213509-tf_efficientdet_d5/model_best.pth.tar --num-classes=4 -b 8 --results rgb_light_fog_moderate.txt --split lightfog --img-size 1280

python validate.py root='' --dataset=seeingthroughfog_rgb_moderate --model tf_efficientdet_d5 --workers 8 --checkpoint output/train/20230521-213509-tf_efficientdet_d5/model_best.pth.tar --num-classes=4 -b 8 --results rgb_dense_fog_moderate.txt --split densefog --img-size 1280

python validate.py root='' --dataset=seeingthroughfog_rgb_moderate --model tf_efficientdet_d5 --workers 8 --checkpoint output/train/20230521-213509-tf_efficientdet_d5/model_best.pth.tar --num-classes=4 -b 8 --results rgb_snow_rain_moderate.txt --split snowrain --img-size 1280




python validate.py root='' --dataset=seeingthroughfog_rgb_hard --model tf_efficientdet_d5 --workers 8 --checkpoint output/train/20230521-213509-tf_efficientdet_d5/model_best.pth.tar --num-classes=4 -b 8 --results rgb_test_clear_hard.txt --split test --img-size 1280

python validate.py root='' --dataset=seeingthroughfog_rgb_hard --model tf_efficientdet_d5 --workers 8 --checkpoint output/train/20230521-213509-tf_efficientdet_d5/model_best.pth.tar --num-classes=4 -b 8 --results rgb_light_fog_hard.txt --split lightfog --img-size 1280

python validate.py root='' --dataset=seeingthroughfog_rgb_hard --model tf_efficientdet_d5 --workers 8 --checkpoint output/train/20230521-213509-tf_efficientdet_d5/model_best.pth.tar --num-classes=4 -b 8 --results rgb_dense_fog_hard.txt --split densefog --img-size 1280

python validate.py root='' --dataset=seeingthroughfog_rgb_hard --model tf_efficientdet_d5 --workers 8 --checkpoint output/train/20230521-213509-tf_efficientdet_d5/model_best.pth.tar --num-classes=4 -b 8 --results rgb_snow_rain_hard.txt --split snowrain --img-size 1280
