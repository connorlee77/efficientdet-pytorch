export CUDA_VISIBLE_DEVICES=1
python validate.py root='' --dataset=seeingthroughfog_rgb_easy --model efficientdetv2_dt --workers 16 --checkpoint output/train/20230524-010421-efficientdetv2_dt/model_best.pth.tar --num-classes=4 -b 16 --results rgb_test_clear_easy.txt --split test --img-size 1280

# python validate.py root='' --dataset=seeingthroughfog_rgb_easy --model tf_efficientdet_d5 --workers 16 --checkpoint output/train/trained_weights/rgb_1280/model_best.pth.tar --num-classes=4 -b 16 --results rgb_test_clear_easy.txt --split test --img-size 1280

# python validate.py root='' --dataset=seeingthroughfog_gated_easy --model efficientdetv2_dt --workers 16 --checkpoint output/train/20230524-010426-efficientdetv2_dt/model_best.pth.tar --num-classes=4 -b 16 --results rgb_test_clear_easy.txt --split test --img-size 1280