export CUDA_VISIBLE_DEVICES=1
python validate.py root='' --dataset=seeingthroughfog_rgb --model efficientdetv2_dt --workers 4 --checkpoint output/train/20230604-155229-efficientdetv2_dt/model_best.pth.tar --num-classes=4 -b 1 --results rgb_test_clear_easy.txt --split test --img-size 1280

# python validate.py root='' --dataset=seeingthroughfog_rgb_easy --model tf_efficientdet_d5 --workers 16 --checkpoint output/train/trained_weights/rgb_1280/model_best.pth.tar --num-classes=4 -b 16 --results rgb_test_clear_easy.txt --split test --img-size 1280

# python validate.py root='' --dataset=seeingthroughfog_gated_easy --model efficientdetv2_dt --workers 16 --checkpoint output/train/20230524-010426-efficientdetv2_dt/model_best.pth.tar --num-classes=4 -b 16 --results rgb_test_clear_easy.txt --split test --img-size 1280