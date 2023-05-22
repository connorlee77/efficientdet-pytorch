import torch
torch.set_float32_matmul_precision('medium')

from stf_dataset import EfficientDetDataModule, SeeingThroughFogDatasetAdaptor, get_train_transforms, get_valid_transforms
from stf_model import EfficientDetModel

import argparse

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning import Trainer
from lightning.pytorch.loggers import WandbLogger


if __name__ == '__main__':

    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')

    parser.add_argument('--batch-size', type=int, default=12)
    parser.add_argument('--devices', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=12)
    parser.add_argument('--num-epochs', type=int, default=100)
    
    parser.add_argument('--dataset', type=str, choices=['cam_stereo_left_rect_aligned', 'gated_full_acc_rect_aligned'])

    args = parser.parse_args()
    print(args)

    if args.dataset == 'cam_stereo_left_rect_aligned':
        modality = 'rgb' 
        mean = [0.26694615, 0.26693442, 0.26698295] 
        std = [0.12035122, 0.12039929, 0.12037755]
    elif args.dataset == 'gated_full_acc_rect_aligned':
        modality = 'gated'
        mean = [0.20945697, 0.20945697, 0.20945697] 
        std = [0.15437697, 0.15437697, 0.15437697]
        
    cars_train_ds = SeeingThroughFogDatasetAdaptor(
        images_dir='/data/SeeingThroughFogDerived/rgb_gated_aligned/{}'.format(args.dataset), 
        labels_dir='/data/SeeingThroughFogDerived/rgb_gated_aligned/labels_aligned', 
        split_file='/data/SeeingThroughFog/splits/train_clear.txt',
        modality=modality,
    )

    cars_val_ds = SeeingThroughFogDatasetAdaptor(
        images_dir='/data/SeeingThroughFogDerived/rgb_gated_aligned/{}'.format(args.dataset), 
        labels_dir='/data/SeeingThroughFogDerived/rgb_gated_aligned/labels_aligned', 
        split_file='/data/SeeingThroughFog/splits/val_clear.txt',
        modality=modality,
    )

    train_transforms = get_train_transforms(target_img_size=1280, modality=modality, mean=mean, std=std)
    valid_transforms = get_valid_transforms(target_img_size=1280, modality=modality, mean=mean, std=std)

    dm = EfficientDetDataModule(
        train_dataset_adaptor=cars_train_ds, 
        validation_dataset_adaptor=cars_val_ds,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        train_transforms=train_transforms,
        valid_transforms=valid_transforms,
    )

    model = EfficientDetModel(
        num_classes=4,
        img_size=1280,
        inference_transforms=valid_transforms,
    )

    wandb_logger = WandbLogger(project="dsf-stf")
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
            monitor="val/loss", 
            every_n_epochs=1,
            save_last=True,
            save_top_k=1,
            mode='min',
            auto_insert_metric_name=True
        )
    lr_monitor = LearningRateMonitor(logging_interval='step')
    trainer = Trainer(
        precision='16-mixed', 
        sync_batchnorm=True,
        callbacks=[checkpoint_callback, lr_monitor], 
        devices=[args.devices], 
        max_epochs=100, 
        num_sanity_val_steps=2, 
        val_check_interval=50, 
        logger=wandb_logger,
    )
    trainer.fit(model, dm)

