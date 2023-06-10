import json
import argparse
import os


def combine_json(json_list):
    new_json = {}
    new_json['type'] = json_list[0]['type']
    new_json['categories'] = json_list[0]['categories']
    new_json['images'] = []
    new_json['annotations'] = []

    for json in json_list:
        new_json['images'].extend(json['images'])
        new_json['annotations'].extend(json['annotations'])

    return new_json
    




if __name__ == '__main__':
    # Take a list of json file path arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_files', nargs='+', help='List of json files to combine')
    parser.add_argument('--out_file', help='Output file name')
    parser.add_argument('--base_path', default=None, help='base path for json files')
    parser.add_argument('--write_path', default=None, help='write path for json files')
    args = parser.parse_args()
    files = args.in_files

    if args.base_path is not None:
        base_path = args.base_path
    else:
        base_path = "/Users/sriadityadeevi/Desktop/Spring Term Academics/Research/Week-10/efficientdet-pytorch/meta/STF/gated"
    if args.write_path is not None:
        write_path = args.write_path
    else:
        write_path = "/Users/sriadityadeevi/Desktop/Spring Term Academics/Research/Week-10/efficientdet-pytorch/meta/temp/correct/gated"

    
    json_list = []
    for file in files:
        json_list.append(json.load(open(os.path.join(base_path, file))))

    new_json = combine_json(json_list)

    with open(os.path.join(write_path, args.out_file), 'w') as outfile:
        json.dump(new_json, outfile)

    print("Done")