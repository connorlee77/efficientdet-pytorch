##########################################
### Ignore ###############################
##########################################

import os
import glob
import tqdm
import json

from sklearn.model_selection import train_test_split

import random
random.seed(0)

def filter_annotation_by_imageID(annotations, image_ids):
    image_ids = set(image_ids)
    ann_keep = []
    for ann in annotations:
        if ann['image_id'] in image_ids:
            ann_keep.append(ann)
    return ann_keep



val_ratio = 0.15
test_ratio = 0.15
train_ratio = 1 - (val_ratio + test_ratio)

json_dir = '/home/connor/repos/efficientdet-pytorch/stf_labels/all'

fog = ['dense_fog_day.json', 'dense_fog_night.json', 'light_fog_day.json', 'light_fog_night.json']
snow = ['snow_day.json', 'snow_night.json']
test_clear = ['test_clear_day.json', 'test_clear_night.json']
train_clear = ['train_clear_day.json', 'train_clear_night.json']
val_clear = ['val_clear_day.json', 'val_clear_night.json']

all_jsons = fog + snow + test_clear + train_clear + val_clear

modality = 'all'

all_images = {
    'train' : [],
    'val' : [],
    'test' : [],
}

for json_name in all_jsons:
    path = os.path.join(json_dir, modality, json_name)
    with open(path, 'r') as f:
        j = json.load(f)
        image_ids = sorted([info['id'] for info in j['images']])

        train_image_ids, val_test_image_ids = train_test_split(image_ids, train_size=train_ratio, shuffle=True, random_state=0)
        val_image_ids, test_image_ids = train_test_split(val_test_image_ids, train_size=0.5, shuffle=True, random_state=0)  
        print(set(train_image_ids).intersection(set(val_image_ids)))
        print(set(test_image_ids).intersection(set(val_image_ids)))

        all_images['train'].append(train_image_ids)
        all_images['val'].append(val_image_ids)
        all_images['test'].append(test_image_ids)
        # Filter json files to create split json files

        train_annotations = [ann ]
