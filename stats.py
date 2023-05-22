import os
import glob
import cv2
import numpy as np
import tqdm


def calc_stats(images_dir, bits):
    mean = np.zeros(3)
    std = np.zeros(3)
    with open('/data/SeeingThroughFog/splits/train_clear.txt', 'r') as f:
        i = 0
        for line in tqdm.tqdm(list(f.readlines())):
            seq, no = line.strip().split(',')
            filename = '{}_{}.tiff'.format(seq, no)
            fp = os.path.join(images_dir, filename)

            img = cv2.imread(fp, -1) / (2**bits - 1)
            mean += img.mean(axis=(0, 1)).squeeze()
            std += img.std(axis=(0, 1)).squeeze()
            i += 1
        
    print(mean / i, std / i)

dataset = 'cam_stereo_left_rect_aligned'
images_dir ='/data/SeeingThroughFogDerived/rgb_gated_aligned/{}'.format(dataset)
bits = 12

calc_stats(images_dir, bits)


dataset = 'gated_full_acc_rect_aligned'
images_dir ='/data/SeeingThroughFogDerived/rgb_gated_aligned/{}'.format(dataset)
bits = 10

calc_stats(images_dir, bits)



