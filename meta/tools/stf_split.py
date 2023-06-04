import json
import argparse
import funcy
from sklearn.model_selection import train_test_split
import numpy as np

def save_coco(file, type_, images, annotations, categories):
    with open(file, 'wt', encoding='UTF-8') as coco:
        json.dump({ 'type': type_, 'images': images, 
            'annotations': annotations, 'categories': categories}, coco, indent=2, sort_keys=True)

def filter_annotations(annotations, images):
    image_ids = funcy.lmap(lambda i: i['id'], images)
    return funcy.lfilter(lambda a: a['image_id'] in image_ids, annotations)


def filter_images(images, annotations):

    annotation_ids = funcy.lmap(lambda i: i['image_id'], annotations)

    return funcy.lfilter(lambda a: a['id'] in annotation_ids, images)

parser = argparse.ArgumentParser(description='Splits COCO annotations file into training and test sets.')
parser.add_argument('annotations', metavar='coco_annotations', type=str,
                    help='Path to COCO annotations file.')
parser.add_argument('sname1', type=str, help='Where to store COCO training annotations')
parser.add_argument('sname2', type=str, help='Where to store COCO test annotations')
parser.add_argument('-s', dest='split', type=float, required=True,
                    help="A percentage of a split; a number in (0, 1)")


args = parser.parse_args()

def main(args):


    # temp_list = args.annotations.split(".")
    temp_list = args.annotations.split("-")
    train_pth = "stf-"+temp_list[1]+"-"+args.sname1+".json"
    test_pth = "stf-"+temp_list[1]+"-"+args.sname2+".json"

    with open(args.annotations, 'rt', encoding='UTF-8') as annotations:
        coco = json.load(annotations)
        type_ = coco['type']
        images = coco['images']
        annotations = coco['annotations']
        categories = coco['categories']

        number_of_images = len(images)

        images_with_annotations = funcy.lmap(lambda a: a['image_id'], annotations)


        images = funcy.lremove(lambda i: i['id'] not in images_with_annotations, images)

        print("Number of images removed: ", number_of_images - len(images))

        print("Number of images after removing images without annotations: ", len(images))


        X_train, X_test = train_test_split(images, train_size=args.split)

        anns_train = filter_annotations(annotations, X_train)
        anns_test=filter_annotations(annotations, X_test)



        save_coco(train_pth, type_, X_train, anns_train, categories)
        save_coco(test_pth, type_, X_test, anns_test, categories)

    with open(train_pth, "r") as read_file:
        train = json.load(read_file)

    with open(test_pth, "r") as read_file:
        test = json.load(read_file)

    with open(args.annotations, "r") as read_file:
        original = json.load(read_file)


    print("Saved {} entries in {} and {} in {}".format(len(anns_train), train_pth, len(anns_test), test_pth))


    # Number and Proportion of Annotations of each category in original

    print("Number of Annotations of each category in original: ")
    prop = []
    for cat in original['categories']:
        count = 0
        for ann in original['annotations']:
            if ann['category_id'] == cat['id']:
                count += 1
        print(cat['name'], ": ", count)
        prop.append(count/len(original['annotations']))

    print("Proportion of Annotations of each category in original: ", prop)


    # Number and Proportion of Annotations of each category in train

    print("Number of Annotations of each category in train: ")
    prop = []
    for cat in train['categories']:
        count = 0
        for ann in train['annotations']:
            if ann['category_id'] == cat['id']:
                count += 1
        print(cat['name'], ": ", count)
        prop.append(count/len(train['annotations']))

    print("Proportion of Annotations of each category in train: ", prop)

    # Number and Proportion of Annotations of each category in test

    print("Number of Annotations of each category in test: ")
    prop = []
    for cat in test['categories']:
        count = 0
        for ann in test['annotations']:
            if ann['category_id'] == cat['id']:
                count += 1
        print(cat['name'], ": ", count)
        prop.append(count/len(test['annotations']))

    print("Proportion of Annotations of each category in test: ", prop)        


if __name__ == "__main__":
    main(args)