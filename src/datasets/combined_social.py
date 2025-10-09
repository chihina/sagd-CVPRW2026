# IMPORTS
import os
from PIL import Image
from glob import glob
from typing import Tuple, Union

import numpy as np
import pandas as pd
import lightning.pytorch as pl

import torch
import torchvision.transforms.functional as TF ###### !!
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torch.utils.data.dataloader import default_collate

from src.utils import Stage, pair ###### !!
from src.transforms import (
    ColorJitter,
    Compose,
    Normalize,
    RandomCropSafeGaze,
    RandomHeadBboxJitter,
    RandomHorizontalFlip,
    Resize,
    ToTensor,
)
from src.datasets.uco_laeo_temporal import VideoLAEODataset_temporal
from src.datasets.videocoatt_temporal import VideoCoAttDataset_temporal
from src.datasets.videoattentiontarget_temporal import VideoAttentionTargetDataset_temporal
from src.datasets.childplay_temporal import ChildPlayDataset_temporal
from src.datasets.gazefollow import GazeFollowDataset


#TODO: update mean/std normalization from GazeFollow to dataset values
IMG_MEAN = [0.44232, 0.40506, 0.36457]
IMG_STD = [0.28674, 0.27776, 0.27995]


# ============================================================================================================ #
#                                              VIDEOCOATT DATA MODULE                                          #
# ============================================================================================================ #
class CombinedSocialDataModule(pl.LightningDataModule):
    def __init__(
        self,
        cfg,
        root_gf: str,
        root_coatt: str,
        root_laeo: str,
        root_vat: str,
        root_childplay: str,
        batch_size: Union[int, dict] = 32,
        num_people: int = 5,
        temporal_context = 2,
        temporal_stride = 3,
        image_size = (224,224)
    ):  
        
        super().__init__()
        self.cfg = cfg
        self.root_gf = root_gf
        self.root_coatt = root_coatt
        self.root_laeo = root_laeo
        self.root_vat = root_vat
        self.root_childplay = root_childplay
        self.num_people = num_people
        self.batch_size = (
            {stage: batch_size for stage in Stage}
            if isinstance(batch_size, int)
            else batch_size
        )
        self.temporal_context = temporal_context
        self.temporal_stride = temporal_stride
        self.image_size = pair(image_size)


    def setup(self, stage: str):
        if stage == "fit":
            ############ Train ##############
            train_transform = Compose(
                [
                    RandomCropSafeGaze(aspect=(self.image_size[0]/self.image_size[1]), p=1.0),
                    # ColorJitter(
                    #     brightness=(0.5, 1.5),
                    #     contrast=(0.5, 1.5),
                    #     saturation=(0.0, 1.5),
                    #     hue=None,
                    #     p=0.8,
                    # ),
                    Resize(img_size=self.image_size, head_size=(224, 224)),
                    ToTensor(),
                    Normalize(
                        img_mean=IMG_MEAN,
                        img_std=IMG_STD
                    ),
                ]
            )
            dataset_coatt = VideoCoAttDataset_temporal(
                cfg=self.cfg,
                root=self.root_coatt, 
                split="train", 
                stride=max(3, self.temporal_context*self.temporal_stride*2),
                transform=train_transform, 
                tr=(-0.1, 0.1), 
                num_people=self.num_people["train"],
                temporal_context=self.temporal_context,
                temporal_stride=self.temporal_stride,
                image_size=self.image_size
            )
            dataset_laeo = VideoLAEODataset_temporal(
                cfg=self.cfg,
                root=self.root_laeo, 
                split="train", 
                stride=max(3, self.temporal_context*self.temporal_stride*2),
                transform=train_transform, 
                tr=(-0.1, 0.1), 
                num_people=self.num_people["train"],
                temporal_context=self.temporal_context,
                temporal_stride=self.temporal_stride,
                image_size=self.image_size
            )
            dataset_vat = VideoAttentionTargetDataset_temporal(
                cfg=self.cfg,
                root=self.root_vat, 
                split="train", 
                stride=max(3, self.temporal_context*self.temporal_stride*2),
                transform=train_transform, 
                tr=(-0.1, 0.1), 
                num_people=self.num_people['train'],
                temporal_context=self.temporal_context,
                temporal_stride=self.temporal_stride,
                image_size=self.image_size
            )
            dataset_childplay = ChildPlayDataset_temporal(
                cfg=self.cfg,
                root=self.root_childplay, 
                split="train", 
                stride=max(3, self.temporal_context*self.temporal_stride*2),
                transform=train_transform, 
                tr=(-0.1, 0.1), 
                num_people=self.num_people['train'],
                temporal_context=self.temporal_context,
                temporal_stride=self.temporal_stride,
                image_size=self.image_size
            )
            dataset_gf = GazeFollowDataset(
                root=self.root_gf, 
                split="train", 
                transform=train_transform, 
                tr=(-0.1, 0.1), 
                num_people=self.num_people["train"]
            )
            
            # self.train_dataset = ConcatDataset([dataset_childplay, dataset_vat, dataset_laeo, dataset_coatt, dataset_gf])
            self.train_dataset = ConcatDataset([dataset_childplay, dataset_vat, dataset_laeo, dataset_coatt])
            # self.train_dataset = ConcatDataset([dataset_vat, dataset_laeo, dataset_coatt])
            # self.train_dataset = ConcatDataset([dataset_coatt])
            # self.train_dataset = ConcatDataset([dataset_laeo])
            # self.train_dataset = ConcatDataset([dataset_vat])
            # self.train_dataset = ConcatDataset([dataset_childplay])

            ########## val #############
            val_transform = Compose(
                [
                    Resize(img_size=self.image_size, head_size=(224, 224)),
                    ToTensor(),
                    Normalize(
                        img_mean=IMG_MEAN,
                        img_std=IMG_STD
                    ),
                ]
            )
                                               
            dataset_coatt = VideoCoAttDataset_temporal(
                cfg=self.cfg,
                root=self.root_coatt, 
                split="val", 
                stride=6,
                transform=val_transform, 
                tr=(0.0, 0.0), 
                num_people=self.num_people["val"],
                temporal_context=self.temporal_context,
                temporal_stride=self.temporal_stride,
                image_size=self.image_size
            )
            dataset_laeo = VideoLAEODataset_temporal(
                cfg=self.cfg,
                root=self.root_laeo, 
                split="val", 
                stride=6,
                transform=val_transform, 
                tr=(0.0, 0.0), 
                num_people=self.num_people["val"],
                temporal_context=self.temporal_context,
                temporal_stride=self.temporal_stride,
                image_size=self.image_size
            )
            dataset_vat = VideoAttentionTargetDataset_temporal(
                cfg=self.cfg,
                root=self.root_vat, 
                split="val", 
                stride=6,
                transform=val_transform, 
                tr=(0.0, 0.0), 
                num_people=self.num_people['val'],
                temporal_context=self.temporal_context,
                temporal_stride=self.temporal_stride,
                image_size=self.image_size
            )
            dataset_childplay = ChildPlayDataset_temporal(
                cfg=self.cfg,
                root=self.root_childplay, 
                split="val", 
                stride=6,
                transform=val_transform, 
                tr=(0.0, 0.0), 
                num_people=self.num_people['val'],
                temporal_context=self.temporal_context,
                temporal_stride=self.temporal_stride,
                image_size=self.image_size
            )
            dataset_gf = GazeFollowDataset(
                root=self.root_gf, 
                split="val", 
                transform=val_transform, 
                tr=(-0.1, 0.1), 
                num_people=self.num_people["val"]
            )
            # self.val_dataset = ConcatDataset([dataset_childplay, dataset_vat, dataset_laeo, dataset_coatt, dataset_gf])
            self.val_dataset = ConcatDataset([dataset_childplay, dataset_vat, dataset_laeo, dataset_coatt])
            # self.val_dataset = ConcatDataset([dataset_vat, dataset_laeo, dataset_coatt])
            # self.val_dataset = ConcatDataset([dataset_coatt])
            # self.val_dataset = ConcatDataset([dataset_laeo])
            # self.val_dataset = ConcatDataset([dataset_vat])
            # self.val_dataset = ConcatDataset([dataset_childplay])

        elif stage == "validate":
            val_transform = Compose(
                [
                    Resize(img_size=self.image_size, head_size=(224, 224)),
                    ToTensor(),
                    Normalize(
                        img_mean=IMG_MEAN,
                        img_std=IMG_STD
                    ),
                ]
            )
                                               
            dataset_coatt = VideoCoAttDataset_temporal(
                cfg=self.cfg,
                root=self.root_coatt, 
                split="val", 
                stride=6,
                transform=val_transform, 
                tr=(0.0, 0.0), 
                num_people=self.num_people["val"],
                temporal_context=self.temporal_context,
                temporal_stride=self.temporal_stride,
                image_size=self.image_size
            )
            dataset_laeo = VideoLAEODataset_temporal(
                root=self.root_laeo, 
                split="val", 
                stride=6,
                transform=val_transform, 
                tr=(0.0, 0.0), 
                num_people=self.num_people["val"],
                temporal_context=self.temporal_context,
                temporal_stride=self.temporal_stride,
                image_size=self.image_size
            )
            dataset_vat = VideoAttentionTargetDataset_temporal(
                root=self.root_vat, 
                split="val", 
                stride=6,
                transform=val_transform, 
                tr=(0.0, 0.0), 
                num_people=self.num_people['val'],
                temporal_context=self.temporal_context,
                temporal_stride=self.temporal_stride,
                image_size=self.image_size
            )
            dataset_childplay = ChildPlayDataset_temporal(
                root=self.root_childplay, 
                split="val", 
                stride=6,
                transform=val_transform, 
                tr=(0.0, 0.0), 
                num_people=self.num_people['val'],
                temporal_context=self.temporal_context,
                temporal_stride=self.temporal_stride,
                image_size=self.image_size
            )
            dataset_gf = GazeFollowDataset(
                root=self.root_gf, 
                split="val", 
                transform=val_transform, 
                tr=(-0.1, 0.1), 
                num_people=self.num_people["val"]
            )
#             self.val_dataset = ConcatDataset([dataset_childplay, dataset_vat, dataset_laeo, dataset_coatt, dataset_gf])
            self.val_dataset = ConcatDataset([dataset_childplay, dataset_vat, dataset_laeo, dataset_coatt])
            # self.val_dataset = ConcatDataset([dataset_vat, dataset_laeo, dataset_coatt])
            # self.val_dataset = ConcatDataset([dataset_coatt])
            # self.val_dataset = ConcatDataset([dataset_laeo])
            # self.val_dataset = ConcatDataset([dataset_vat])
            # self.val_dataset = ConcatDataset([dataset_childplay])

        elif stage == "test":
            aspect = False    # maintain aspect ratio
            if aspect:
                img_size = self.image_size[1]
            else:
                img_size = self.image_size
            test_transform = Compose(
                [
                    Resize(img_size=img_size, head_size=(224, 224)),
                    ToTensor(),
                    Normalize(
                        img_mean=IMG_MEAN,
                        img_std=IMG_STD
                    ),
                ]
            )
            dataset_coatt = VideoCoAttDataset_temporal(
                cfg=self.cfg,
                root=self.root_coatt, 
                split="test", 
                stride=3,
                transform=test_transform, 
                tr=(0.0, 0.0), 
                num_people=self.num_people["test"],
                temporal_context=self.temporal_context,
                temporal_stride=self.temporal_stride,
                image_size=self.image_size,
                aspect = aspect
            )
            dataset_laeo = VideoLAEODataset_temporal(
                cfg=self.cfg,
                root=self.root_laeo, 
                split="test", 
                stride=3,
                transform=test_transform, 
                tr=(0.0, 0.0), 
                num_people=self.num_people["test"],
                temporal_context=self.temporal_context,
                temporal_stride=self.temporal_stride,
                image_size=self.image_size,
                aspect = aspect
            )
            dataset_vat = VideoAttentionTargetDataset_temporal(
                cfg=self.cfg,
                root=self.root_vat, 
                split="test", 
                stride=3,
                transform=test_transform, 
                tr=(0.0, 0.0), 
                num_people=self.num_people['test'],
                temporal_context=self.temporal_context,
                temporal_stride=self.temporal_stride,
                image_size=self.image_size,
                aspect = aspect
            )
            dataset_childplay = ChildPlayDataset_temporal(
                cfg=self.cfg,
                root=self.root_childplay, 
                split="test", 
                stride=3,
                transform=test_transform, 
                tr=(0.0, 0.0), 
                num_people=self.num_people['test'],
                temporal_context=self.temporal_context,
                temporal_stride=self.temporal_stride,
                image_size=self.image_size,
                aspect = aspect
            )
            dataset_gf = GazeFollowDataset(
                root=self.root_gf, 
                split="test", 
                transform=test_transform, 
                tr=(-0.1, 0.1), 
                num_people=self.num_people["test"]
            )
#             self.test_dataset = ConcatDataset([dataset_childplay, dataset_vat, dataset_laeo, dataset_coatt, dataset_gf])
            self.test_dataset = ConcatDataset([dataset_childplay, dataset_vat, dataset_laeo, dataset_coatt])
            # self.test_dataset = ConcatDataset([dataset_vat, dataset_laeo, dataset_coatt])
            # self.test_dataset = ConcatDataset([dataset_coatt])
            # self.test_dataset = ConcatDataset([dataset_laeo])
            # self.test_dataset = ConcatDataset([dataset_vat])
            # self.test_dataset = ConcatDataset([dataset_childplay])

        elif stage == "predict":
            predict_transform = Compose(
                [
                    Resize(img_size=224, head_size=(224, 224)),
                    ToTensor(),
                    Normalize(
                        img_mean=IMG_MEAN,
                        img_std=IMG_STD
                    ),
                ]
            )
            self.predict_dataset = VideoCoAttDataset_temporal(
                cfg=self.cfg,
                root=self.root, 
                split="test",
                stride=1,
                transform=predict_transform, 
                tr=(0.0, 0.0), 
                num_people=self.num_people["test"]
            )
        
        # randomly sample a few examples for quick debugging
        if True:
            max_num = 200
            if stage == "fit":
                self.train_dataset = torch.utils.data.Subset(self.train_dataset, list(range(0, max_num)))
                self.val_dataset = torch.utils.data.Subset(self.val_dataset, list(range(0, max_num)))
            elif stage == "validate":
                self.val_dataset = torch.utils.data.Subset(self.val_dataset, list(range(0, max_num)))
            elif stage == "test":
                self.test_dataset = torch.utils.data.Subset(self.test_dataset, list(range(0, max_num)))
            elif stage == "predict":
                self.predict_dataset = torch.utils.data.Subset(self.predict_dataset, list(range(0, max_num)))

    def train_dataloader(self):
        dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size[Stage.TRAIN],
            shuffle=True,
            num_workers=14,
            pin_memory=False,
            # pin_memory=True,
            # collate_fn=self.custom_collate_fn
        )
        return dataloader

    def val_dataloader(self):
        dataloader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size[Stage.VAL],
            shuffle=False,
            num_workers=6,
            pin_memory=False,
            # pin_memory=True,
            # collate_fn=self.custom_collate_fn
        )
        return dataloader

    def test_dataloader(self):
        dataloader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size[Stage.TEST],
            shuffle=True,
            num_workers=6,
            pin_memory=False,
            # pin_memory=True,
            # collate_fn=self.custom_collate_fn
        )
        return dataloader

    def predict_dataloader(self):
        dataloader = DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size[Stage.PREDICT],
            shuffle=False,
            num_workers=8,
            pin_memory=False,
            # pin_memory=True,
            # collate_fn=self.custom_collate_fn
        )
        return dataloader
    
    # def custom_collate_fn(self, batch):
    #     pop_keys = []
    #     for item in batch[0].keys():
    #         if type(item) != torch.Tensor and type(item) != np.ndarray:
    #             pop_keys.append(item)

    #     pop_items = {}
    #     for key in pop_keys:
    #         pop_items[key] = [item.pop(key) for item in batch]

    #     batch = default_collate(batch)
    #     for key in pop_keys:
    #         batch[key] = pop_items[key]

    #     return batch