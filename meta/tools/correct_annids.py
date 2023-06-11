import json
import argparse
import os
from copy import deepcopy




def correct_json(json_list):

    new_json_list = []

    for json in json_list:

        new_json = {}
        new_json['type'] = deepcopy(json['type'])
        new_json['categories'] = deepcopy(json['categories'])
        new_json['images'] = deepcopy(json['images'])
        new_json['annotations'] = deepcopy(json['annotations'])

        for n, ann in enumerate(new_json['annotations']):
            ann['id'] = n+1

        new_json_list.append(new_json)

    
    return new_json_list

        

if __name__ == '__main__':
    # Take a list of json file path arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_path', default=None, help='base path for json files')

    args = parser.parse_args()

    if args.base_path is not None:
        base_path = args.base_path

    else:
        base_path = "/Users/sriadityadeevi/Desktop/Spring Term Academics/Research/Week-10/efficientdet-pytorch/meta/STF/gated"

    json_file_names = [filename for filename in os.listdir(base_path) if filename.endswith('.json')]

    json_list = []
    for file in json_file_names:
        json_list.append(json.load(open(os.path.join(base_path, file))))

    new_json_list = correct_json(json_list)


    for n,file in enumerate(json_file_names):

        os.remove(os.path.join(base_path, file))
        print("Removed file: ", file)

        with open(os.path.join(base_path, file), 'w') as outfile:
            json.dump(new_json_list[n], outfile)

        print("Written file: ", file)

    print("Done")
    

