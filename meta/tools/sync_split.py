import json
import argparse
import os
import funcy

def sync_split(json_dict, write_path):

    new_json_dict = {}
        
    for file in json_dict.keys():
        new_json_dict[file] = {}
        new_json_dict[file]['type'] = json_dict[file]['type']
        new_json_dict[file]['categories'] = json_dict[file]['categories']

        comp_json = json.load(open(os.path.join(write_path, file.split("-")[1]+".json")))

        print("Using the path ", os.path.join(write_path, file.split("-")[1]+".json"))

        ref_image_ids = funcy.lmap(lambda i: i['id'], json_dict[file]['images'])

        ref_ann_ids = funcy.lmap(lambda a: a['image_id'], json_dict[file]['annotations'])

        new_json_dict[file]['images'] = funcy.lremove(lambda i: i['id'] not in ref_image_ids, comp_json['images'])
        print("Number of images removed: ", len(json_dict[file]['images']) - len(new_json_dict[file]['images']))
        new_json_dict[file]['annotations'] = funcy.lremove(lambda a: a['image_id'] not in ref_ann_ids, comp_json['annotations'])
        print("Number of annotations removed: ", len(json_dict[file]['annotations']) - len(new_json_dict[file]['annotations']))


    return new_json_dict









if __name__ == '__main__':
    # Take a list of json file path arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_files', nargs='+', help='List of json files to combine')
    parser.add_argument('--base_path', default=None, help='base path for json files')
    parser.add_argument('--write_path', default=None, help='write path for json files')
    args = parser.parse_args()

    files = args.in_files

    if args.base_path is not None:
        base_path = args.base_path
    else:
        base_path = "/Users/sriadityadeevi/Desktop/Spring Term Academics/Research/Week-10/efficientdet-pytorch/meta/all"

    if args.write_path is not None:
        write_path = args.write_path
    else:
        write_path = "/Users/sriadityadeevi/Desktop/Spring Term Academics/Research/Week-10/efficientdet-pytorch/meta/rgb"


    json_dict = {}

    for file in files:
        json_dict[file] = json.load(open(os.path.join(base_path, file)))


    new_json_dict = sync_split(json_dict, write_path)


    for file in new_json_dict.keys():
        with open(os.path.join(write_path, file), 'w') as outfile:
            json.dump(new_json_dict[file], outfile)


    print("Done")


    