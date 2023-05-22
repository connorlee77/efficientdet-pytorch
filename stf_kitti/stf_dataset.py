import os
import numpy as np
import cv2

import torch
from torch.utils.data import Dataset
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

def decode_visible_labels(value):
    if value == "True":
        return True
    elif value == "False":
        return False
    else:
        return None

class SeeingThroughFogDatasetAdaptor:

    # visible_idx: rgb-23, gated-24, lidar-25, radar-26
    def __init__(self, images_dir, labels_dir, split_file, modality):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        

        self.images = []
        with open(split_file, 'r') as f:
            for line in f.readlines():
                seq, no = line.strip().split(',')
                filename = '{}_{}'.format(seq, no)
                self.images.append(filename)

        if modality == 'rgb':
            self.bits = 12
            self.visible_idx = 23
        elif modality == 'gated':
            self.bits = 10
            self.visible_idx = 24
        else:
            raise Exception('Not a valid visible modality: {}'.format(modality))

        self.label_name_idx_map = {
            'DontCare' : -1, 
            'train' : -1,
            'Obstacle' : -1,

            'LargeVehicle' : 1, 
            'LargeVehicle_is_group' : 1, 
            
            'Pedestrian' : 2,
            'person' : 2,
            'Pedestrian_is_group' : 2, 
            
            'PassengerCar' : 3, 
            'PassengerCar_is_group' : 3, 
            'Vehicle' : 3, 
            'Vehicle_is_group' : 3, 

            'RidableVehicle' : 4,
            'RidableVehicle_is_group' : 4, 
        }

    def __len__(self) -> int:
        return len(self.images)

    def get_image_and_labels_by_idx(self, index):
        name = self.images[index]

        image_path = os.path.join(self.images_dir, '{}.tiff'.format(name))
        label_path = os.path.join(self.labels_dir, '{}.txt'.format(name))
        assert os.path.exists(image_path), image_path
        assert os.path.exists(label_path), label_path

        image = cv2.imread(image_path, -1) / (2**self.bits - 1)
        if len(image.shape) < 3:
            image = np.stack([image]*3, axis=2)

        bboxes = []
        class_labels = []
        with open(label_path, 'r') as f:
            for line in f.readlines():
                kitti_properties = line.strip().split()

                visible = decode_visible_labels(kitti_properties[self.visible_idx]),
                if not visible:
                    continue

                bbox_class = kitti_properties[0]
                bbox_idx = self.label_name_idx_map[bbox_class]

                xmin = int(round(float(kitti_properties[4])))
                ymin = int(round(float(kitti_properties[5])))
                xmax = int(round(float(kitti_properties[6])))
                ymax = int(round(float(kitti_properties[7])))

                if (xmax - xmin) <= 1 or (ymax - ymin) <= 1:
                    continue

                bbox = np.array([xmin, ymin, xmax, ymax])
                bboxes.append(bbox)
                class_labels.append(bbox_idx)
        
        if bboxes:
            bboxes = np.array(bboxes, ndmin=2, dtype=np.float32)
            class_labels = np.array(class_labels, dtype=np.int64)
        else:
            bboxes = np.zeros((0, 4), dtype=np.float32)
            class_labels = np.array([], dtype=np.int64)

        return image, bboxes, class_labels, index







def get_train_transforms(target_img_size=512, modality='rgb', mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.LongestMaxSize(target_img_size, p=1),
            A.PadIfNeeded(target_img_size, target_img_size, border_mode=cv2.BORDER_CONSTANT, value=0),
            A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], max_pixel_value=1.0),
            ToTensorV2(p=1),
        ],
        p=1.0,
        bbox_params=A.BboxParams(
            format="pascal_voc", min_area=0, min_visibility=0, label_fields=["labels"]
        )
    )


def get_valid_transforms(target_img_size=512, modality='rgb', mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    return A.Compose(
        [
            A.LongestMaxSize(target_img_size, p=1),
            A.PadIfNeeded(target_img_size, target_img_size, border_mode=cv2.BORDER_CONSTANT, value=0),
            A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], max_pixel_value=1.0),
            ToTensorV2(p=1),
        ],
        p=1.0,
        bbox_params=A.BboxParams(
            format="pascal_voc", min_area=0, min_visibility=0, label_fields=["labels"]
        )
    )


class EfficientDetDataset(Dataset):
    def __init__(
        self, dataset_adaptor, transforms=get_valid_transforms()
    ):
        self.ds = dataset_adaptor
        self.transforms = transforms

    def __getitem__(self, index):
        (
            image,
            pascal_bboxes,
            class_labels,
            image_id,
        ) = self.ds.get_image_and_labels_by_idx(index)

        sample = {
            "image": np.array(image, dtype=np.float32),
            "bboxes": pascal_bboxes,
            "labels": class_labels,
        }

        sample = self.transforms(**sample)
        if len(sample["bboxes"]) == 0:
            sample["bboxes"].append([0, 0, 0, 0])
            sample["labels"].append(-1)
        sample["bboxes"] = np.array(sample["bboxes"])
        image = sample["image"]
        labels = sample["labels"]

        _, new_h, new_w = image.shape
        
        sample["bboxes"][:, [0, 1, 2, 3]] = sample["bboxes"][
            :, [1, 0, 3, 2]
        ]  # convert to yxyx

        target = {
            "bboxes": torch.as_tensor(sample["bboxes"], dtype=torch.float32),
            "labels": torch.as_tensor(labels),
            "image_id": torch.tensor([image_id]),
            "img_size": (new_h, new_w),
            "img_scale": torch.tensor([1.0]),
        }

        return image, target, image_id

    def __len__(self):
        return len(self.ds)


class EfficientDetDataModule(LightningDataModule):
    
    def __init__(self,
                train_dataset_adaptor,
                validation_dataset_adaptor,
                train_transforms=get_train_transforms(target_img_size=1280),
                valid_transforms=get_valid_transforms(target_img_size=1280),
                num_workers=4,
                batch_size=8):
        
        self.train_ds = train_dataset_adaptor
        self.valid_ds = validation_dataset_adaptor
        self.train_tfms = train_transforms
        self.valid_tfms = valid_transforms
        self.num_workers = num_workers
        self.batch_size = batch_size
        super().__init__()

    def train_dataset(self) -> EfficientDetDataset:
        return EfficientDetDataset(
            dataset_adaptor=self.train_ds, transforms=self.train_tfms
        )

    def train_dataloader(self) -> DataLoader:
        train_dataset = self.train_dataset()
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

        return train_loader

    def val_dataset(self) -> EfficientDetDataset:
        return EfficientDetDataset(
            dataset_adaptor=self.valid_ds, transforms=self.valid_tfms
        )

    def val_dataloader(self) -> DataLoader:
        valid_dataset = self.val_dataset()
        valid_loader = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            drop_last=True,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

        return valid_loader
    
    @staticmethod
    def collate_fn(batch):
        images, targets, image_ids = tuple(zip(*batch))
        images = torch.stack(images)
        images = images.float()

        boxes = [target["bboxes"].float() for target in targets]
        labels = [target["labels"].float() for target in targets]
        img_size = torch.tensor([target["img_size"] for target in targets]).float()
        img_scale = torch.tensor([target["img_scale"] for target in targets]).float()

        annotations = {
            "bbox": boxes,
            "cls": labels,
            "img_size": img_size,
            "img_scale": img_scale,
        }

        return images, annotations, targets, image_ids