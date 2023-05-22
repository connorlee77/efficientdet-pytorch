from numbers import Number
from typing import List
from functools import singledispatch

import numpy as np
import torch

from fastcore.dispatch import typedispatch
from pytorch_lightning import LightningModule

from ensemble_boxes import ensemble_boxes_wbf


from effdet.config.model_config import efficientdet_model_param_dict
from effdet import get_efficientdet_config, EfficientDet, DetBenchTrain
from effdet.efficientdet import HeadNet
from effdet.config.model_config import efficientdet_model_param_dict

from stf_dataset import get_valid_transforms, get_train_transforms

def create_model(num_classes=4, image_size=1280, architecture="efficientdetv2_dt"):
    
    config = get_efficientdet_config(architecture)
    config.update({'num_classes': num_classes})
    config.update({'image_size': (image_size, image_size)})
    
    print(config)

    net = EfficientDet(config, pretrained_backbone=True)
    net.class_net = HeadNet(
        config,
        num_outputs=config.num_classes,
    )
    return DetBenchTrain(net, config)


def run_wbf(predictions, image_size=512, iou_thr=0.44, skip_box_thr=0.43, weights=None):
    bboxes = []
    confidences = []
    class_labels = []

    for prediction in predictions:
        boxes = [(prediction["boxes"] / image_size).tolist()]
        scores = [prediction["scores"].tolist()]
        labels = [prediction["classes"].tolist()]

        boxes, scores, labels = ensemble_boxes_wbf.weighted_boxes_fusion(
            boxes,
            scores,
            labels,
            weights=weights,
            iou_thr=iou_thr,
            skip_box_thr=skip_box_thr,
        )
        boxes = boxes * (image_size - 1)
        bboxes.append(boxes.tolist())
        confidences.append(scores.tolist())
        class_labels.append(labels.tolist())

    return bboxes, confidences, class_labels


class EfficientDetModel(LightningModule):

    display_ids = {-1 : 'Dont Care', 1 : 'Large Vehicle', 2 : 'Person', 3 : 'Car', 4 : 'Ridable Vehicle'}

    def __init__(
        self,
        num_classes=1,
        img_size=1280,
        prediction_confidence_threshold=0.3,
        learning_rate=1e-3,
        wbf_iou_threshold=0.44,
        inference_transforms=get_valid_transforms(target_img_size=1280),
        model_architecture='efficientdetv2_dt',
    ):
        super().__init__()
        self.img_size = img_size
        self.model = create_model(
            num_classes, img_size, architecture=model_architecture
        )
        self.prediction_confidence_threshold = prediction_confidence_threshold
        self.lr = learning_rate
        self.wbf_iou_threshold = wbf_iou_threshold
        self.inference_tfms = inference_transforms
        
    def forward(self, images, targets):
        return self.model(images, targets)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
        return {
           'optimizer': optimizer,
           'lr_scheduler': scheduler, # Changed scheduler to lr_scheduler
        }

    def training_step(self, batch, batch_idx):
        images, annotations, _, image_ids = batch

        losses = self.model(images, annotations)
        if batch_idx % 10 == 0:
            
            mean = torch.as_tensor(np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)).to(self.device)
            std = torch.as_tensor(np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)).to(self.device)
            img = images[0] * std + mean
            log_annotations = annotations['bbox'][0].cpu().numpy().astype(int)
            log_classes = annotations['cls'][0].cpu().numpy().astype(int)
            boxes = {
                "prediction" : {
                    "box_data" : [{
                        "position" :{
                            "minX" : int(box[1]),
                            "minY" : int(box[0]),
                            "maxX" : int(box[3]),
                            "maxY" : int(box[2]),
                        },
                        "class_id" : int(clas),
                        "box_caption" : '{}'.format(self.display_ids[clas]),
                        "domain" : "pixel"
                    } 
                    for box, clas in zip(log_annotations, log_classes)
                    ],
                }
            }

            self.logger.log_image('train/image', [img], boxes=[boxes])

        self.log("train/loss", losses["loss"])
        self.log("train/class_loss", losses["class_loss"])
        self.log("train/box_loss", losses["box_loss"])

        return losses['loss']


    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        images, annotations, targets, image_ids = batch
        outputs = self.model(images, annotations)

        # [x_min, y_min, x_max, y_max, score, class]
        detections = outputs["detections"]
        if batch_idx % 5 == 0:
            
            mean = torch.as_tensor(np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)).to(self.device)
            std = torch.as_tensor(np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)).to(self.device)
            img = images[0] * std + mean
            log_annotations = detections[0,:,:4].cpu().numpy().astype(int)
            log_classes = detections[0,:,5].cpu().numpy().astype(int)
            log_conf = detections[0,:,4].cpu().numpy()
            boxes = {
                "prediction" : {
                    "box_data" : [{
                        "position" :{
                            "minX" : int(box[0]),
                            "minY" : int(box[1]),
                            "maxX" : int(box[2]),
                            "maxY" : int(box[3]),
                        },
                        "class_id" : int(clas),
                        "box_caption" : '{}: {:.2f}'.format(self.display_ids[clas], conf),
                        "domain" : "pixel",
                        "scores" : { "score" : float(conf)}
                    } 
                    for box, conf, clas in zip(log_annotations, log_conf, log_classes)
                    ],
                }
            }

            self.logger.log_image('val/image', [img], boxes=[boxes])

        batch_predictions = {
            "predictions": detections,
            "targets": targets,
            "image_ids": image_ids,
        }

        logging_losses = {
            "class_loss": outputs["class_loss"].detach(),
            "box_loss": outputs["box_loss"].detach(),
        }

        self.log("val/loss", outputs["loss"])
        self.log("val/class_loss", logging_losses["class_loss"])
        self.log("val/box_loss", logging_losses["box_loss"])

        return {'loss': outputs["loss"], 'batch_predictions': batch_predictions}
    
    
    @typedispatch
    def predict(self, images: List):
        """
        For making predictions from images
        Args:
            images: a list of PIL images

        Returns: a tuple of lists containing bboxes, predicted_class_labels, predicted_class_confidences

        """
        image_sizes = [(image.size[1], image.size[0]) for image in images]
        images_tensor = torch.stack(
            [
                self.inference_tfms(
                    image=np.array(image, dtype=np.float32),
                    labels=np.ones(1),
                    bboxes=np.array([[0, 0, 1, 1]]),
                )["image"]
                for image in images
            ]
        )

        return self._run_inference(images_tensor, image_sizes)

    @typedispatch
    def predict(self, images_tensor: torch.Tensor):
        """
        For making predictions from tensors returned from the model's dataloader
        Args:
            images_tensor: the images tensor returned from the dataloader

        Returns: a tuple of lists containing bboxes, predicted_class_labels, predicted_class_confidences

        """
        if images_tensor.ndim == 3:
            images_tensor = images_tensor.unsqueeze(0)
        if (
            images_tensor.shape[-1] != self.img_size
            or images_tensor.shape[-2] != self.img_size
        ):
            raise ValueError(
                f"Input tensors must be of shape (N, 3, {self.img_size}, {self.img_size})"
            )

        num_images = images_tensor.shape[0]
        image_sizes = [(self.img_size, self.img_size)] * num_images

        return self._run_inference(images_tensor, image_sizes)

    def _run_inference(self, images_tensor, image_sizes):
        dummy_targets = self._create_dummy_inference_targets(
            num_images=images_tensor.shape[0]
        )

        detections = self.model(images_tensor.to(self.device), dummy_targets)[
            "detections"
        ]
        (
            predicted_bboxes,
            predicted_class_confidences,
            predicted_class_labels,
        ) = self.post_process_detections(detections)

        scaled_bboxes = self.__rescale_bboxes(
            predicted_bboxes=predicted_bboxes, image_sizes=image_sizes
        )

        return scaled_bboxes, predicted_class_labels, predicted_class_confidences
    
    def _create_dummy_inference_targets(self, num_images):
        dummy_targets = {
            "bbox": [
                torch.tensor([[0.0, 0.0, 0.0, 0.0]], device=self.device)
                for i in range(num_images)
            ],
            "cls": [torch.tensor([1.0], device=self.device) for i in range(num_images)],
            "img_size": torch.tensor(
                [(self.img_size, self.img_size)] * num_images, device=self.device
            ).float(),
            "img_scale": torch.ones(num_images, device=self.device).float(),
        }

        return dummy_targets
    
    def post_process_detections(self, detections):
        predictions = []
        for i in range(detections.shape[0]):
            predictions.append(
                self._postprocess_single_prediction_detections(detections[i])
            )

        predicted_bboxes, predicted_class_confidences, predicted_class_labels = run_wbf(
            predictions, image_size=self.img_size, iou_thr=self.wbf_iou_threshold
        )

        return predicted_bboxes, predicted_class_confidences, predicted_class_labels

    def _postprocess_single_prediction_detections(self, detections):
        boxes = detections.detach().cpu().numpy()[:, :4]
        scores = detections.detach().cpu().numpy()[:, 4]
        classes = detections.detach().cpu().numpy()[:, 5]
        indexes = np.where(scores > self.prediction_confidence_threshold)[0]
        boxes = boxes[indexes]

        return {"boxes": boxes, "scores": scores[indexes], "classes": classes[indexes]}

    def __rescale_bboxes(self, predicted_bboxes, image_sizes):
        scaled_bboxes = []
        for bboxes, img_dims in zip(predicted_bboxes, image_sizes):
            im_h, im_w = img_dims

            if len(bboxes) > 0:
                scaled_bboxes.append(
                    (
                        np.array(bboxes)
                        * [
                            im_w / self.img_size,
                            im_h / self.img_size,
                            im_w / self.img_size,
                            im_h / self.img_size,
                        ]
                    ).tolist()
                )
            else:
                scaled_bboxes.append(bboxes)

        return scaled_bboxes