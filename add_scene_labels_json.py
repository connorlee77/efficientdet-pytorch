import os
import json
import glob

fps = glob.glob('stf_jsons_with_scene_idx/all/all/*.json')

new_fps = []
for f in fps:
    if 'rain' not in f:
        new_fps.append(f)
fps = new_fps



save_folder = 'stf_jsons_with_scene_idx_fix/all/all'
os.makedirs(save_folder, exist_ok=True)

split_files = glob.glob('/data/SeeingThroughFog/splits/*_day.txt') + glob.glob('/data/SeeingThroughFog/splits/*_night.txt')

scene_dict = {
    'test_clear_day' : 1, 
    'test_clear_night' : 2,

    'val_clear_day' : 1, 
    'val_clear_night' : 2, 

    'train_clear_day' : 1, 
    'train_clear_night' : 2, 

    'dense_fog_day' : 3, 
    'dense_fog_night' : 4,

    'light_fog_day' : 3, 
    'light_fog_night' : 4,
    
    'snow_day' : 5, 
    'snow_night' : 6,

    # 'rain' : 7,
}


def create_scene_mapping():
    mapping = {}
    for fp in split_files:
        split = os.path.basename(fp).replace('.txt', '')
        assert split in scene_dict, split
        idx = scene_dict[split]
        with open(fp, 'r') as f:
            for line in f.readlines():
                date, no = line.strip().split(',')
                id = '{}_{}'.format(date.strip(), no.strip())
                assert id not in mapping
                mapping[id] = idx
    return mapping

mapping = create_scene_mapping()


for filepath in fps:
    save_name = os.path.join(save_folder, os.path.basename(filepath))
    
    data = None
    with open(filepath, 'r') as f:
        data = json.load(f)
        image_list = data['images']
        print(filepath)
        for entry in image_list:
            id = entry['id']
            scene_id = mapping[id]
            entry['scene'] = scene_id

    with open(save_name, 'w') as f:
        print(data['images'])
        json.dump(data, f)