import json
import os 
import glob
import tqdm

files = glob.glob('meta/STF/rgb/stf-full-*.json')

sets = []
for fp in files:
    with open(fp, 'r') as f:
        j = json.load(f)
        
        ids = set()
        for ann in tqdm.tqdm(j['annotations']):
            ids.add(ann['image_id'])
        sets.append(ids)

for s1 in sets:
    p = ''
    for s2 in sets:
        common = set.intersection(s1, s2)
        p += str(len(common)).zfill(4) + ' '
    
    print(p)



# files = glob.glob('stf_labels/all/rgb/*.json')
files = sorted([
    'stf_labels/all/rgb/train_clear_day.json', 
    'stf_labels/all/rgb/dense_fog_night.json', 
    'stf_labels/all/rgb/light_fog_night.json', 
    'stf_labels/all/rgb/val_clear_day.json', 
    'stf_labels/all/rgb/test_clear_night.json', 
    'stf_labels/all/rgb/test_clear_day.json', 
    'stf_labels/all/rgb/train_clear_night.json', 
    'stf_labels/all/rgb/val_clear_night.json', 
    # 'stf_labels/all/rgb/rain.json', 
    'stf_labels/all/rgb/snow_day.json', 
    'stf_labels/all/rgb/snow_night.json', 
    'stf_labels/all/rgb/dense_fog_day.json', 
    'stf_labels/all/rgb/light_fog_day.json',
])

sets = []
for fp in files:
    with open(fp, 'r') as f:
        j = json.load(f)
        
        ids = set()
        for ann in tqdm.tqdm(j['annotations']):
            ids.add(ann['image_id'])
        sets.append(ids)

for s1 in sets:
    p = ''
    for s2 in sets:
        common = set.intersection(s1, s2)
        p += str(len(common)).zfill(4) + ' '
    
    print(p)