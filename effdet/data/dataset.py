""" Detection dataset

Hacked together by Ross Wightman
"""
import torch.utils.data as data
import numpy as np
import os
from PIL import Image
from .parsers import create_parser
import cv2

import effdet.data.cv2_transforms as cv2_transforms

class DetectionDatset(data.Dataset):
    """`Object Detection Dataset. Use with parsers for COCO, VOC, and OpenImages.
    Args:
        parser (string, Parser):
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``

    """

    def __init__(self, data_dir, parser=None, parser_kwargs=None, transform=None):
        super(DetectionDatset, self).__init__()
        parser_kwargs = parser_kwargs or {}
        self.data_dir = data_dir
        if isinstance(parser, str):
            self._parser = create_parser(parser, **parser_kwargs)
        else:
            assert parser is not None and len(parser.img_ids)
            self._parser = parser
        self._transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, annotations (target)).
        """
        img_info = self._parser.img_infos[index]
        if 'scene' in img_info:
            target = dict(img_idx=index, img_size=(img_info['width'], img_info['height']), img_scene=img_info['scene'])
        else:
            target = dict(img_idx=index, img_size=(img_info['width'], img_info['height']))
        if self._parser.has_labels:
            ann = self._parser.get_ann_info(index)
            target.update(ann)

        img_path = self.data_dir / img_info['file_name']
        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img, target = self.transform(img, target)
        
        # draw_img = img.transpose(1, 2, 0)
        # for box in target['bbox']:
        #     y1, x1, y2, x2 = box.astype(int)
        #     cv2.rectangle(draw_img, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # cv2.imwrite('testimg.png', draw_img)
        return img, target

    def __len__(self):
        return len(self._parser.img_ids)

    @property
    def parser(self):
        return self._parser

    @property
    def transform(self):
        return self._transform

    @transform.setter
    def transform(self, t):
        self._transform = t

stf_map = {
    1 : 'LargeVehicle',
    2 : 'Person',
    3 : 'Car',
    4 : 'Bike',
}

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
class XBitDetectionDatset(data.Dataset):
    """`Object Detection Dataset. Use with parsers for COCO, VOC, and OpenImages.
    Args:
        parser (string, Parser):
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``

    """

    def __init__(self, 
                 data_dir, 
                 parser=None, parser_kwargs=None, transform=None, 
                 mode='train', bits=16, 
                 mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD):
        super(XBitDetectionDatset, self).__init__()
        parser_kwargs = parser_kwargs or {}
        self.data_dir = data_dir
        if isinstance(parser, str):
            self._parser = create_parser(parser, **parser_kwargs)
        else:
            assert parser is not None and len(parser.img_ids)
            self._parser = parser
        
        self.mode = mode
        self.bits = bits

        if mode == 'train':
            self.fixed_transform = cv2_transforms.transforms_coco_train(
                1280,
                use_prefetcher=True,
                mean=mean,
                std=std,
            )
        else:
            self.fixed_transform = cv2_transforms.transforms_coco_eval(
                1280,
                use_prefetcher=True,
                mean=mean,
                std=std,
            )

        self._transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, annotations (target)).
        """
        img_info = self._parser.img_infos[index]
        target = dict(img_idx=index, img_size=(img_info['width'], img_info['height']))
        if self._parser.has_labels:
            ann = self._parser.get_ann_info(index)
            target.update(ann)

        img_path = os.path.join(self.data_dir, img_info['file_name'])
        img = cv2.imread(img_path, -1) / (2**self.bits - 1) * 255
        if len(img.shape) < 3:
            img = np.stack([img]*3, axis=2)
        else:
            img = img[:,:,::-1]
        img, target = self.fixed_transform(img, target)
        # print(target)
        # draw_img = img.transpose(1, 2, 0)
        # draw_img = cv2.imread(img_path, -1) / (2**self.bits - 1) * 255
        # for box, cls in zip(target['bbox'], target['cls']):
        #     if cls >= 1: 
        #         class_id = stf_map[cls]
        #         y1, x1, y2, x2 = box.astype(int)
        #         cv2.putText(
        #             draw_img, 
        #             '{}'.format(class_id), 
        #             (x1, y1 - 10), 
        #             fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        #             fontScale=1,
        #             color=(255, 0, 255),
        #             thickness=2,
        #         )
                
        #         cv2.rectangle(draw_img, (x1, y1), (x2, y2), (255, 0, 255), 2)

        # os.makedirs('debug_val', exist_ok=True)
        # cv2.imwrite('debug_val/{}.png'.format(img_info['file_name'].replace('tiff', '')), draw_img)
        # exit(0)
        return img, target

    def __len__(self):
        return len(self._parser.img_ids)

    @property
    def parser(self):
        return self._parser

    @property
    def transform(self):
        return self._transform

    @transform.setter
    def transform(self, t):
        self._transform = t



class SkipSubset(data.Dataset):
    r"""
    Subset of a dataset at specified indices.

    Arguments:
        dataset (Dataset): The whole Dataset
        n (int): skip rate (select every nth)
    """
    def __init__(self, dataset, n=2):
        self.dataset = dataset
        assert n >= 1
        self.indices = np.arange(len(dataset))[::n]

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)

    @property
    def parser(self):
        return self.dataset.parser

    @property
    def transform(self):
        return self.dataset.transform

    @transform.setter
    def transform(self, t):
        self.dataset.transform = t
