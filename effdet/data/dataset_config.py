""" COCO, VOC, OpenImages dataset configurations

Copyright 2020 Ross Wightman
"""
import os
from dataclasses import dataclass, field
from typing import Dict

# This is where all the data is. Place json folder in this directory
SEEING_THROUGH_FOG_DATA_DIR = '/data/SeeingThroughFogDerived/rgb_gated_aligned/'
FLIR_ALIGNED_DATA_DIR = "/home/ganlu/workspace/FLIR_Aligned/"


@dataclass
class CocoCfg:
    variant: str = None
    parser: str = 'coco'
    num_classes: int = 80
    splits: Dict[str, dict] = None

@dataclass
class SeeingThroughFogRGBCfg(CocoCfg):
    variant: str = 'rgb'
    splits: Dict[str, dict] = field(default_factory=lambda: dict(
        train=dict(
            ann_filename=os.path.join(SEEING_THROUGH_FOG_DATA_DIR, 'stf_coco_no_person/all/rgb/train_clear.json'), 
            img_dir=os.path.join(SEEING_THROUGH_FOG_DATA_DIR, 'cam_stereo_left_rect_aligned'), has_labels=True),
        val=dict(
            ann_filename=os.path.join(SEEING_THROUGH_FOG_DATA_DIR, 'stf_coco_no_person/all/rgb/val_clear.json'), 
            img_dir=os.path.join(SEEING_THROUGH_FOG_DATA_DIR, 'cam_stereo_left_rect_aligned'), has_labels=True),
        test=dict(
            ann_filename=os.path.join(SEEING_THROUGH_FOG_DATA_DIR, 'stf_coco_no_person/all/rgb/test_clear.json'), 
            img_dir=os.path.join(SEEING_THROUGH_FOG_DATA_DIR, 'cam_stereo_left_rect_aligned'), has_labels=True),
    ))

@dataclass
class SeeingThroughFogRGBEasyCfg(CocoCfg):
    variant: str = 'rgb'
    splits: Dict[str, dict] = field(default_factory=lambda: dict(
        train=dict(
            ann_filename=os.path.join(SEEING_THROUGH_FOG_DATA_DIR, 'stf_coco_json_0idx_DontCare/easy/all/train_clear.json'), 
            img_dir=os.path.join(SEEING_THROUGH_FOG_DATA_DIR, 'cam_stereo_left_rect_aligned'), has_labels=True),
        val=dict(
            ann_filename=os.path.join(SEEING_THROUGH_FOG_DATA_DIR, 'stf_coco_json_0idx_DontCare/easy/all/val_clear.json'), 
            img_dir=os.path.join(SEEING_THROUGH_FOG_DATA_DIR, 'cam_stereo_left_rect_aligned'), has_labels=True),
        # test=dict(
        #     ann_filename=os.path.join(SEEING_THROUGH_FOG_DATA_DIR, 'stf_coco_json_0idx_DontCare/easy/all/test_clear.json'), 
        #     img_dir=os.path.join(SEEING_THROUGH_FOG_DATA_DIR, 'cam_stereo_left_rect_aligned'), has_labels=True),
        test=dict(
            ann_filename='/home/connor/cv_conversions/stf_labels/easy/all/test_clear.json', 
            img_dir=os.path.join(SEEING_THROUGH_FOG_DATA_DIR, 'cam_stereo_left_rect_aligned'), has_labels=True),
        lightfog=dict(
            ann_filename=os.path.join(SEEING_THROUGH_FOG_DATA_DIR, 'stf_coco_json_0idx_DontCare/easy/all/light_fog.json'), 
            img_dir=os.path.join(SEEING_THROUGH_FOG_DATA_DIR, 'cam_stereo_left_rect_aligned'), has_labels=True),
        densefog=dict(
            ann_filename=os.path.join(SEEING_THROUGH_FOG_DATA_DIR, 'stf_coco_json_0idx_DontCare/easy/all/dense_fog.json'), 
            img_dir=os.path.join(SEEING_THROUGH_FOG_DATA_DIR, 'cam_stereo_left_rect_aligned'), has_labels=True),
        snowrain=dict(
            ann_filename=os.path.join(SEEING_THROUGH_FOG_DATA_DIR, 'stf_coco_json_0idx_DontCare/easy/all/snow_rain.json'), 
            img_dir=os.path.join(SEEING_THROUGH_FOG_DATA_DIR, 'cam_stereo_left_rect_aligned'), has_labels=True),
    ))

@dataclass
class SeeingThroughFogRGBMediumCfg(CocoCfg):
    variant: str = 'rgb'
    splits: Dict[str, dict] = field(default_factory=lambda: dict(
        train=dict(
            ann_filename=os.path.join(SEEING_THROUGH_FOG_DATA_DIR, 'stf_coco_json_0idx_DontCare/moderate/all/train_clear.json'), 
            img_dir=os.path.join(SEEING_THROUGH_FOG_DATA_DIR, 'cam_stereo_left_rect_aligned'), has_labels=True),
        val=dict(
            ann_filename=os.path.join(SEEING_THROUGH_FOG_DATA_DIR, 'stf_coco_json_0idx_DontCare/moderate/all/val_clear.json'), 
            img_dir=os.path.join(SEEING_THROUGH_FOG_DATA_DIR, 'cam_stereo_left_rect_aligned'), has_labels=True),
        test=dict(
            ann_filename=os.path.join(SEEING_THROUGH_FOG_DATA_DIR, 'stf_coco_json_0idx_DontCare/moderate/all/test_clear.json'), 
            img_dir=os.path.join(SEEING_THROUGH_FOG_DATA_DIR, 'cam_stereo_left_rect_aligned'), has_labels=True),
        lightfog=dict(
            ann_filename=os.path.join(SEEING_THROUGH_FOG_DATA_DIR, 'stf_coco_json_0idx_DontCare/moderate/all/light_fog.json'), 
            img_dir=os.path.join(SEEING_THROUGH_FOG_DATA_DIR, 'cam_stereo_left_rect_aligned'), has_labels=True),
        densefog=dict(
            ann_filename=os.path.join(SEEING_THROUGH_FOG_DATA_DIR, 'stf_coco_json_0idx_DontCare/moderate/all/dense_fog.json'), 
            img_dir=os.path.join(SEEING_THROUGH_FOG_DATA_DIR, 'cam_stereo_left_rect_aligned'), has_labels=True),
        snowrain=dict(
            ann_filename=os.path.join(SEEING_THROUGH_FOG_DATA_DIR, 'stf_coco_json_0idx_DontCare/moderate/all/snow_rain.json'), 
            img_dir=os.path.join(SEEING_THROUGH_FOG_DATA_DIR, 'cam_stereo_left_rect_aligned'), has_labels=True),
    ))

@dataclass
class SeeingThroughFogRGBHardCfg(CocoCfg):
    variant: str = 'rgb'
    splits: Dict[str, dict] = field(default_factory=lambda: dict(
        train=dict(
            ann_filename=os.path.join(SEEING_THROUGH_FOG_DATA_DIR, 'stf_coco_json_0idx_DontCare/hard/all/train_clear.json'), 
            img_dir=os.path.join(SEEING_THROUGH_FOG_DATA_DIR, 'cam_stereo_left_rect_aligned'), has_labels=True),
        val=dict(
            ann_filename=os.path.join(SEEING_THROUGH_FOG_DATA_DIR, 'stf_coco_json_0idx_DontCare/hard/all/val_clear.json'), 
            img_dir=os.path.join(SEEING_THROUGH_FOG_DATA_DIR, 'cam_stereo_left_rect_aligned'), has_labels=True),
        test=dict(
            ann_filename=os.path.join(SEEING_THROUGH_FOG_DATA_DIR, 'stf_coco_json_0idx_DontCare/hard/all/test_clear.json'), 
            img_dir=os.path.join(SEEING_THROUGH_FOG_DATA_DIR, 'cam_stereo_left_rect_aligned'), has_labels=True),
        lightfog=dict(
            ann_filename=os.path.join(SEEING_THROUGH_FOG_DATA_DIR, 'stf_coco_json_0idx_DontCare/hard/all/light_fog.json'), 
            img_dir=os.path.join(SEEING_THROUGH_FOG_DATA_DIR, 'cam_stereo_left_rect_aligned'), has_labels=True),
        densefog=dict(
            ann_filename=os.path.join(SEEING_THROUGH_FOG_DATA_DIR, 'stf_coco_json_0idx_DontCare/hard/all/dense_fog.json'), 
            img_dir=os.path.join(SEEING_THROUGH_FOG_DATA_DIR, 'cam_stereo_left_rect_aligned'), has_labels=True),
        snowrain=dict(
            ann_filename=os.path.join(SEEING_THROUGH_FOG_DATA_DIR, 'stf_coco_json_0idx_DontCare/hard/all/snow_rain.json'), 
            img_dir=os.path.join(SEEING_THROUGH_FOG_DATA_DIR, 'cam_stereo_left_rect_aligned'), has_labels=True),
    ))

@dataclass
class SeeingThroughFogGatedCfg(CocoCfg):
    variant: str = 'gated'
    splits: Dict[str, dict] = field(default_factory=lambda: dict(
        train=dict(
            ann_filename=os.path.join(SEEING_THROUGH_FOG_DATA_DIR, 'stf_coco_no_person/all/gated/train_clear.json'), 
            img_dir=os.path.join(SEEING_THROUGH_FOG_DATA_DIR, 'gated_full_acc_rect_aligned'), has_labels=True),
        val=dict(
            ann_filename=os.path.join(SEEING_THROUGH_FOG_DATA_DIR, 'stf_coco_no_person/all/gated/val_clear.json'), 
            img_dir=os.path.join(SEEING_THROUGH_FOG_DATA_DIR, 'gated_full_acc_rect_aligned'), has_labels=True),
        test=dict(
            ann_filename=os.path.join(SEEING_THROUGH_FOG_DATA_DIR, 'stf_coco_no_person/all/gated/test_clear.json'), 
            img_dir=os.path.join(SEEING_THROUGH_FOG_DATA_DIR, 'gated_full_acc_rect_aligned'), has_labels=True),
    ))

@dataclass
class SeeingThroughFogGatedEasyCfg(CocoCfg):
    variant: str = 'gated'
    splits: Dict[str, dict] = field(default_factory=lambda: dict(
        train=dict(
            ann_filename=os.path.join(SEEING_THROUGH_FOG_DATA_DIR, 'stf_coco_json_0idx_DontCare/easy/all/train_clear.json'), 
            img_dir=os.path.join(SEEING_THROUGH_FOG_DATA_DIR, 'gated_full_acc_rect_aligned'), has_labels=True),
        val=dict(
            ann_filename=os.path.join(SEEING_THROUGH_FOG_DATA_DIR, 'stf_coco_json_0idx_DontCare/easy/all/val_clear.json'), 
            img_dir=os.path.join(SEEING_THROUGH_FOG_DATA_DIR, 'gated_full_acc_rect_aligned'), has_labels=True),
        # test=dict(
        #     ann_filename=os.path.join(SEEING_THROUGH_FOG_DATA_DIR, 'stf_coco_json_0idx_DontCare/easy/all/test_clear.json'), 
        #     img_dir=os.path.join(SEEING_THROUGH_FOG_DATA_DIR, 'gated_full_acc_rect_aligned'), has_labels=True),
        test=dict(
            ann_filename='/home/connor/cv_conversions/stf_labels/easy/all/test_clear.json', 
            img_dir=os.path.join(SEEING_THROUGH_FOG_DATA_DIR, 'gated_full_acc_rect_aligned'), has_labels=True),
        lightfog=dict(
            ann_filename=os.path.join(SEEING_THROUGH_FOG_DATA_DIR, 'stf_coco_json_0idx_DontCare/easy/all/light_fog.json'), 
            img_dir=os.path.join(SEEING_THROUGH_FOG_DATA_DIR, 'gated_full_acc_rect_aligned'), has_labels=True),
        densefog=dict(
            ann_filename=os.path.join(SEEING_THROUGH_FOG_DATA_DIR, 'stf_coco_json_0idx_DontCare/easy/all/dense_fog.json'), 
            img_dir=os.path.join(SEEING_THROUGH_FOG_DATA_DIR, 'gated_full_acc_rect_aligned'), has_labels=True),
        snowrain=dict(
            ann_filename=os.path.join(SEEING_THROUGH_FOG_DATA_DIR, 'stf_coco_json_0idx_DontCare/easy/all/snow_rain.json'), 
            img_dir=os.path.join(SEEING_THROUGH_FOG_DATA_DIR, 'gated_full_acc_rect_aligned'), has_labels=True),
    ))

@dataclass
class SeeingThroughFogGatedMediumCfg(CocoCfg):
    variant: str = 'gated'
    splits: Dict[str, dict] = field(default_factory=lambda: dict(
        train=dict(
            ann_filename=os.path.join(SEEING_THROUGH_FOG_DATA_DIR, 'stf_coco_json_0idx_DontCare/moderate/all/train_clear.json'), 
            img_dir=os.path.join(SEEING_THROUGH_FOG_DATA_DIR, 'gated_full_acc_rect_aligned'), has_labels=True),
        val=dict(
            ann_filename=os.path.join(SEEING_THROUGH_FOG_DATA_DIR, 'stf_coco_json_0idx_DontCare/moderate/all/val_clear.json'), 
            img_dir=os.path.join(SEEING_THROUGH_FOG_DATA_DIR, 'gated_full_acc_rect_aligned'), has_labels=True),
        test=dict(
            ann_filename=os.path.join(SEEING_THROUGH_FOG_DATA_DIR, 'stf_coco_json_0idx_DontCare/moderate/all/test_clear.json'), 
            img_dir=os.path.join(SEEING_THROUGH_FOG_DATA_DIR, 'gated_full_acc_rect_aligned'), has_labels=True),
        lightfog=dict(
            ann_filename=os.path.join(SEEING_THROUGH_FOG_DATA_DIR, 'stf_coco_json_0idx_DontCare/moderate/all/light_fog.json'), 
            img_dir=os.path.join(SEEING_THROUGH_FOG_DATA_DIR, 'gated_full_acc_rect_aligned'), has_labels=True),
        densefog=dict(
            ann_filename=os.path.join(SEEING_THROUGH_FOG_DATA_DIR, 'stf_coco_json_0idx_DontCare/moderate/all/dense_fog.json'), 
            img_dir=os.path.join(SEEING_THROUGH_FOG_DATA_DIR, 'gated_full_acc_rect_aligned'), has_labels=True),
        snowrain=dict(
            ann_filename=os.path.join(SEEING_THROUGH_FOG_DATA_DIR, 'stf_coco_json_0idx_DontCare/moderate/all/snow_rain.json'), 
            img_dir=os.path.join(SEEING_THROUGH_FOG_DATA_DIR, 'gated_full_acc_rect_aligned'), has_labels=True),
    ))

@dataclass
class SeeingThroughFogGatedHardCfg(CocoCfg):
    variant: str = 'gated'
    splits: Dict[str, dict] = field(default_factory=lambda: dict(
        train=dict(
            ann_filename=os.path.join(SEEING_THROUGH_FOG_DATA_DIR, 'stf_coco_json_0idx_DontCare/hard/all/train_clear.json'), 
            img_dir=os.path.join(SEEING_THROUGH_FOG_DATA_DIR, 'gated_full_acc_rect_aligned'), has_labels=True),
        val=dict(
            ann_filename=os.path.join(SEEING_THROUGH_FOG_DATA_DIR, 'stf_coco_json_0idx_DontCare/hard/all/val_clear.json'), 
            img_dir=os.path.join(SEEING_THROUGH_FOG_DATA_DIR, 'gated_full_acc_rect_aligned'), has_labels=True),
        test=dict(
            ann_filename=os.path.join(SEEING_THROUGH_FOG_DATA_DIR, 'stf_coco_json_0idx_DontCare/hard/all/test_clear.json'), 
            img_dir=os.path.join(SEEING_THROUGH_FOG_DATA_DIR, 'gated_full_acc_rect_aligned'), has_labels=True),
        lightfog=dict(
            ann_filename=os.path.join(SEEING_THROUGH_FOG_DATA_DIR, 'stf_coco_json_0idx_DontCare/hard/all/light_fog.json'), 
            img_dir=os.path.join(SEEING_THROUGH_FOG_DATA_DIR, 'gated_full_acc_rect_aligned'), has_labels=True),
        densefog=dict(
            ann_filename=os.path.join(SEEING_THROUGH_FOG_DATA_DIR, 'stf_coco_json_0idx_DontCare/hard/all/dense_fog.json'), 
            img_dir=os.path.join(SEEING_THROUGH_FOG_DATA_DIR, 'gated_full_acc_rect_aligned'), has_labels=True),
        snowrain=dict(
            ann_filename=os.path.join(SEEING_THROUGH_FOG_DATA_DIR, 'stf_coco_json_0idx_DontCare/hard/all/snow_rain.json'), 
            img_dir=os.path.join(SEEING_THROUGH_FOG_DATA_DIR, 'gated_full_acc_rect_aligned'), has_labels=True),
    ))


@dataclass
class M3fdRGBCfg(CocoCfg):
    variant: str = ''
    splits: Dict[str, dict] = field(default_factory=lambda: dict(
        train=dict(ann_filename='/home/carson/data/m3fd/meta/m3fd-train.json', img_dir='/home/carson/data/m3fd/Vis', has_labels=True),
        val=dict(ann_filename='/home/carson/data/m3fd/meta/m3fd-val.json', img_dir='/home/carson/data/m3fd/Vis', has_labels=True),
        test=dict(ann_filename='/home/carson/data/m3fd/meta/m3fd-test.json', img_dir='/home/carson/data/m3fd/Vis', has_labels=True)
    ))

@dataclass
class M3fdThermalCfg(CocoCfg):
    variant: str = ''
    splits: Dict[str, dict] = field(default_factory=lambda: dict(
        train=dict(ann_filename='/home/carson/data/m3fd/meta/m3fd-train.json', img_dir='/home/carson/data/m3fd/Ir', has_labels=True),
        val=dict(ann_filename='/home/carson/data/m3fd/meta/m3fd-val.json', img_dir='/home/carson/data/m3fd/Ir', has_labels=True),
        test=dict(ann_filename='/home/carson/data/m3fd/meta/m3fd-test.json', img_dir='/home/carson/data/m3fd/Ir', has_labels=True)
    ))

@dataclass
class FlirV2Cfg(CocoCfg):
    variant: str = ''
    splits: Dict[str, dict] = field(default_factory=lambda: dict(
        train=dict(ann_filename='/home/carson/data/FLIR_ADAS_v2/images_thermal_train/coco.json', img_dir='/home/carson/data/FLIR_ADAS_v2/images_thermal_train', has_labels=True),
        val=dict(ann_filename='/home/carson/data/FLIR_ADAS_v2/images_thermal_val/coco.json', img_dir='/home/carson/data/FLIR_ADAS_v2/images_thermal_val', has_labels=True),
        test=dict(ann_filename='/home/carson/data/FLIR_ADAS_v2/video_thermal_test/coco.json', img_dir='/home/carson/data/FLIR_ADAS_v2/video_thermal_test', has_labels=True),
    ))

@dataclass
class FlirV2RGBCfg(CocoCfg):
    variant: str = ''
    splits: Dict[str, dict] = field(default_factory=lambda: dict(
        train=dict(ann_filename='/home/carson/data/FLIR_ADAS_v2/images_rgb_train/coco.json', img_dir='/home/carson/data/FLIR_ADAS_v2/images_rgb_train', has_labels=True),
        val=dict(ann_filename='/home/carson/data/FLIR_ADAS_v2/images_rgb_val/coco.json', img_dir='/home/carson/data/FLIR_ADAS_v2/images_rgb_val', has_labels=True),
        test=dict(ann_filename='/home/carson/data/FLIR_ADAS_v2/video_rgb_test/coco.json', img_dir='/home/carson/data/FLIR_ADAS_v2/video_rgb_test', has_labels=True),
    ))

@dataclass
class FlirAlignedThermalCfg(CocoCfg):
    variant: str = ''
    splits: Dict[str, dict] = field(default_factory=lambda: dict(
        train=dict(
            ann_filename=os.path.join(FLIR_ALIGNED_DATA_DIR, 'images_thermal_train/flir_train.json'), 
            img_dir=os.path.join(FLIR_ALIGNED_DATA_DIR, 'images_thermal_train/data'), has_labels=True),
        val=dict(
            ann_filename=os.path.join(FLIR_ALIGNED_DATA_DIR, 'images_thermal_train/flir_val.json'), 
            img_dir=os.path.join(FLIR_ALIGNED_DATA_DIR, 'images_thermal_train/data'), has_labels=True),
        test=dict(
            ann_filename=os.path.join(FLIR_ALIGNED_DATA_DIR, 'images_thermal_test/flir.json'), 
            img_dir=os.path.join(FLIR_ALIGNED_DATA_DIR, 'images_thermal_test/data'), has_labels=True)
    ))

@dataclass
class FlirAlignedRGBCfg(CocoCfg):
    variant: str = ''
    splits: Dict[str, dict] = field(default_factory=lambda: dict(
        train=dict(
            ann_filename=os.path.join(FLIR_ALIGNED_DATA_DIR, 'images_rgb_train/flir_train.json'), 
            img_dir=os.path.join(FLIR_ALIGNED_DATA_DIR, 'images_rgb_train/data'), has_labels=True),
        val=dict(
            ann_filename=os.path.join(FLIR_ALIGNED_DATA_DIR, 'images_rgb_train/flir_val.json'), 
            img_dir=os.path.join(FLIR_ALIGNED_DATA_DIR, 'images_rgb_train/data'), has_labels=True),
        test=dict(
            ann_filename=os.path.join(FLIR_ALIGNED_DATA_DIR, 'images_rgb_test/flir.json'), 
            img_dir=os.path.join(FLIR_ALIGNED_DATA_DIR, 'images_rgb_test/data'), has_labels=True)
    ))


@dataclass
class Coco2017Cfg(CocoCfg):
    variant: str = '2017'
    splits: Dict[str, dict] = field(default_factory=lambda: dict(
        train=dict(ann_filename='annotations/instances_train2017.json', img_dir='train2017', has_labels=True),
        val=dict(ann_filename='annotations/instances_val2017.json', img_dir='val2017', has_labels=True),
        test=dict(ann_filename='annotations/image_info_test2017.json', img_dir='test2017', has_labels=False),
        testdev=dict(ann_filename='annotations/image_info_test-dev2017.json', img_dir='test2017', has_labels=False),
    ))


@dataclass
class Coco2014Cfg(CocoCfg):
    variant: str = '2014'
    splits: Dict[str, dict] = field(default_factory=lambda: dict(
        train=dict(ann_filename='annotations/instances_train2014.json', img_dir='train2014', has_labels=True),
        val=dict(ann_filename='annotations/instances_val2014.json', img_dir='val2014', has_labels=True),
        test=dict(ann_filename='', img_dir='test2014', has_labels=False),
    ))


@dataclass
class VocCfg:
    variant: str = None
    parser: str = 'voc'
    num_classes: int = 80
    img_filename: str = '%s.jpg'
    splits: Dict[str, dict] = None


@dataclass
class Voc2007Cfg(VocCfg):
    variant: str = '2007'
    splits: Dict[str, dict] = field(default_factory=lambda: dict(
        train=dict(
            split_filename='VOC2007/ImageSets/Main/train.txt',
            ann_filename='VOC2007/Annotations/%s.xml',
            img_dir='VOC2007/JPEGImages', ),
        val=dict(
            split_filename='VOC2007/ImageSets/Main/val.txt',
            ann_filename='VOC2007/Annotations/%s.xml',
            img_dir='VOC2007/JPEGImages'),
        #test=dict(img_dir='JPEGImages')
    ))


@dataclass
class Voc2012Cfg(VocCfg):
    variant: str = '2012'
    splits: Dict[str, dict] = field(default_factory=lambda: dict(
        train=dict(
            split_filename='VOC2012/ImageSets/Main/train.txt',
            ann_filename='VOC2012/Annotations/%s.xml',
            img_dir='VOC2012/JPEGImages'),
        val=dict(
            split_filename='VOC2012/ImageSets/Main/val.txt',
            ann_filename='VOC2012/Annotations/%s.xml',
            img_dir='VOC2012/JPEGImages'),
        #test=dict(img_dir='JPEGImages', split_file=None)
    ))


@dataclass
class Voc0712Cfg(VocCfg):
    variant: str = '0712'
    splits: Dict[str, dict] = field(default_factory=lambda: dict(
        train=dict(
            split_filename=['VOC2007/ImageSets/Main/trainval.txt', 'VOC2012/ImageSets/Main/trainval.txt'],
            ann_filename=['VOC2007/Annotations/%s.xml', 'VOC2012/Annotations/%s.xml'],
            img_dir=['VOC2007/JPEGImages', 'VOC2012/JPEGImages']),
        val=dict(
            split_filename='VOC2007/ImageSets/Main/test.txt',
            ann_filename='VOC2007/Annotations/%s.xml',
            img_dir='VOC2007/JPEGImages'),
        #test=dict(img_dir='JPEGImages', split_file=None)
    ))



@dataclass
class OpenImagesCfg:
    variant: str = None
    parser: str = 'openimages'
    num_classes: int = None
    img_filename = '%s.jpg'
    splits: Dict[str, dict] = None


@dataclass
class OpenImagesObjCfg(OpenImagesCfg):
    num_classes: int = 601
    categories_map: str = 'annotations/class-descriptions-boxable.csv'


@dataclass
class OpenImagesSegCfg(OpenImagesCfg):
    num_classes: int = 350
    categories_map: str = 'annotations/classes-segmentation.txt'


@dataclass
class OpenImagesObjV5Cfg(OpenImagesObjCfg):
    splits: Dict[str, dict] = field(default_factory=lambda: dict(
        train=dict(
            img_dir='train', img_info='annotations/train-info.csv', has_labels=True, prefix_levels=1,
            ann_bbox='annotations/train-annotations-bbox.csv',
            ann_img_label='annotations/train-annotations-human-imagelabels-boxable.csv',
        ),
        val=dict(
            img_dir='validation', img_info='annotations/validation-info.csv', has_labels=True, prefix_levels=0,
            ann_bbox='annotations/validation-annotations-bbox.csv',
            ann_img_label='annotations/validation-annotations-human-imagelabels-boxable.csv',
        ),
        test=dict(
            img_dir='test', img_info='', has_labels=True, prefix_levels=0,
            ann_bbox='annotations/test-annotations-bbox.csv',
            ann_img_label='annotations/test-annotations-human-imagelabels-boxable.csv',
        )
    ))


@dataclass
class OpenImagesObjChallenge2019Cfg(OpenImagesObjCfg):
    num_classes: int = 500
    categories_map: str = 'annotations/challenge-2019/challenge-2019-classes-description-500.csv'
    splits: Dict[str, dict] = field(default_factory=lambda: dict(
        train=dict(
            img_dir='train', img_info='annotations/train-info.csv', has_labels=True, prefix_levels=1,
            ann_bbox='annotations/challenge-2019/challenge-2019-train-detection-bbox.csv',
            ann_img_label='annotations/challenge-2019/challenge-2019-train-detection-human-imagelabels.csv',
        ),
        val=dict(
            img_dir='validation', img_info='annotations/validation-info.csv', has_labels=True, prefix_levels=0,
            ann_bbox='annotations/challenge-2019/challenge-2019-validation-detection-bbox.csv',
            ann_img_label='annotations/challenge-2019/challenge-2019-validation-detection-human-imagelabels.csv',
        ),
        test=dict(
            img_dir='challenge2019', img_info='annotations/challenge-2019/challenge2019-info', prefix_levels=0,
            has_labels=False, ann_bbox='', ann_img_label='',
        )
    ))


@dataclass
class OpenImagesSegV5Cfg(OpenImagesSegCfg):
    num_classes: int = 300
    splits: Dict[str, dict] = field(default_factory=lambda: dict(
        train=dict(),
        val=dict(),
        test=dict()
    ))


@dataclass
class OpenImagesSegChallenge2019Cfg(OpenImagesSegCfg):
    num_classes: int = 300
    ann_class_map: str = 'annotations/challenge-2019/challenge-2019-classes-description-segmentable.csv'
    splits: Dict[str, dict] = field(default_factory=lambda: dict(
        train=dict(),
        val=dict(),
        test=dict()
    ))
