# ************************************************************************ #
#                                 IMPORTS                                  #
# ************************************************************************ #


import os
import random
import sys
from glob import glob
from typing import Dict, List, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import lightning.pytorch as pl
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.ops import box_iou

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
from src.utils import Stage, expand_bbox


IMG_MEAN = [0.44232, 0.40506, 0.36457]
IMG_STD = [0.28674, 0.27776, 0.27995]
class ACQDIVDataset(Dataset):
    def __init__(
        self,
        root: str,
        split: str = "train",
        stride: int = 3,
        transform: Union[Compose, None] = None,
        tr=(-0.1, 0.1),
        num_people=5
    ):
        super().__init__()
        self.root = root
        self.split = "validate" if split == "val" else split
        self.stride = stride
        self.jitter_bbox = RandomHeadBboxJitter(p=1.0, tr=tr)
        self.transform = transform
        self.num_people = num_people
        self.annotations, self.paths, self.path2indices = self.load_annotations()
    

    def load_annotations(self):
        annotation_file = os.path.join(self.root, f"testingData_path.csv")
        annotations = pd.read_csv(annotation_file)
        path2indices = annotations.groupby("path").indices
        
        # Discard images with only 1 person annotation
        mask = (annotations.groupby("path").size() < 2)
        paths_to_del = mask[mask].index.tolist()
        for path in paths_to_del:
            indices = path2indices[path]
            annotations = annotations.drop(indices, axis=0)
                    
        # Get unique paths
        paths = annotations.path.unique()
        path2indices = annotations.groupby("path").indices
        
        # sample frames depending on stride
        sampled_indices = np.arange(len(paths), step=self.stride)
        paths = paths[sampled_indices]
        self.length = len(paths)

        return annotations, paths, path2indices

    
    def __getitem__(self, index):

        path = self.paths[index]
        indices = self.path2indices[path]
        img_annotations = self.annotations.iloc[indices]
        
        # Load Image
        image = Image.open(os.path.join(self.root, "testing_data", path)).convert("RGB")
        img_w, img_h = image.size
        
        # Get person head bounding boxes
        head_bboxes = img_annotations[["head_bbox_x_min", "head_bbox_y_min", "head_bbox_x_max", "head_bbox_y_max"]]
        head_bboxes = torch.from_numpy(head_bboxes.values.astype(np.float32))        
        if self.split == "train":
            head_bboxes = self.jitter_bbox(head_bboxes, img_w, img_h) 
        num_heads = len(head_bboxes)
        num_missing_heads = max(self.num_people - num_heads, 0) if self.num_people != "all" else 0

        # Extract Heads
        heads = []
        for head_bbox in head_bboxes:
            heads.append(image.crop(head_bbox.int().tolist()))
        
        # Create (Normalized) Gaze Points
        gaze_pts = torch.from_numpy(img_annotations[["gaze_x", "gaze_y"]].values.astype(np.float32))
        gaze_pts[gaze_pts[:, 0] != -1.] /= torch.tensor([img_w, img_h])
        
        # Shuffle inputs during training
        if self.split == "train":
            rand_indices = torch.randperm(head_bboxes.size(0))
            head_bboxes = head_bboxes[rand_indices]
            heads = [heads[i] for i in rand_indices]
            gaze_pts = gaze_pts[rand_indices]
            
        # Select the first `num_people` people   
        num_keep = num_heads if self.num_people == "all" else self.num_people
        head_bboxes = head_bboxes[:num_keep]
        heads = heads[:num_keep]
        gaze_pts = gaze_pts[:num_keep]
        num_valid_heads = len(heads)
            
        # Pad missing people (ie. heads, head_bboxes, gaze_pt and coatt)
        if num_missing_heads > 0:
            head_bboxes = torch.cat([torch.zeros((num_missing_heads, 4)), head_bboxes])
            heads = num_missing_heads * [Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))] + heads
            gaze_pts = torch.cat([torch.zeros((num_missing_heads, 2)), gaze_pts])

        # Normalize Head Bboxes
        head_bboxes /= torch.tensor([img_w, img_h, img_w, img_h], dtype=float)

        # Compute Head Bbox Centers
        head_centers = torch.hstack([
            (head_bboxes[:, [0]] + head_bboxes[:, [2]]) / 2, 
            (head_bboxes[:, [1]] + head_bboxes[:, [3]]) / 2
        ])    
            
        # Build Sample
        sample = {
                "image": image,
                "heads": heads,
                "head_bboxes": head_bboxes,
                "gaze_pts": gaze_pts,
                "gaze_vecs": F.normalize(gaze_pts - head_centers, p=2, dim=1),
                "num_valid_people": num_valid_heads,
                "img_size": torch.tensor((img_w, img_h), dtype=torch.long),
                "path": path
        }
        
        # Transform
        if self.transform:
            sample = self.transform(sample)

        return sample   
    
    def __len__(self):
        return self.length

    
# ============================================================================================================ #
#                                              ACQDIV DATA MODULE                                          #
# ============================================================================================================ #
class ACQDIVDataModule(pl.LightningDataModule):
    def __init__(
        self,
        root: str,
        batch_size: Union[int, dict] = 32,
        num_people: int = 5
    ):  
        
        super().__init__()
        self.root = root
        self.num_people = num_people
        self.batch_size = (
            {stage: batch_size for stage in Stage}
            if isinstance(batch_size, int)
            else batch_size
        )


    def setup(self, stage: str):
        if stage == "fit":
            train_transform = Compose(
                [
                    RandomCropSafeGaze(aspect=1.0, p=1.0),
                    RandomHorizontalFlip(p=0.5),
                    ColorJitter(
                        brightness=(0.5, 1.5),
                        contrast=(0.5, 1.5),
                        saturation=(0.0, 1.5),
                        hue=None,
                        p=0.8,
                    ),
                    Resize(img_size=(224, 224), head_size=(224, 224)),
                    ToTensor(),
                    Normalize(
                        img_mean=IMG_MEAN,
                        img_std=IMG_STD
                    ),
                ]
            )
            self.train_dataset = ACQDIVDataset(
                root=self.root, 
                split="train", 
                stride=1,
                transform=train_transform, 
                tr=(-0.1, 0.1), 
                num_people=self.num_people["train"]
            )

            val_transform = Compose(
                [
                    Resize(img_size=224, head_size=(224, 224)),
                    ToTensor(),
                    Normalize(
                        img_mean=IMG_MEAN,
                        img_std=IMG_STD
                    ),
                ]
            )
            self.val_dataset = ACQDIVDataset(
                root=self.root, 
                split="val", 
                stride=1,
                transform=val_transform, 
                tr=(0.0, 0.0), 
                num_people=self.num_people["val"]
            )

        elif stage == "validate":
            val_transform = Compose(
                [
                    Resize(img_size=224, head_size=(224, 224)),
                    ToTensor(),
                    Normalize(
                        img_mean=IMG_MEAN,
                        img_std=IMG_STD
                    ),
                ]
            )
            self.val_dataset = ACQDIVDataset(
                root=self.root, 
                split="val",
                stride=1,
                transform=val_transform, 
                tr=(0.0, 0.0), 
                num_people=self.num_people["val"]
            )

        elif stage == "test":
            test_transform = Compose(
                [
                    Resize(img_size=224, head_size=(224, 224)),
                    ToTensor(),
                    Normalize(
                        img_mean=IMG_MEAN,
                        img_std=IMG_STD
                    ),
                ]
            )
            self.test_dataset = ACQDIVDataset(
                root=self.root, 
                split="test", 
                stride=1,
                transform=test_transform, 
                tr=(0.0, 0.0), 
                num_people=self.num_people["test"]
            )
            

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
            self.predict_dataset = ACQDIVDataset(
                root=self.root, 
                split="test",
                stride=1,
                transform=predict_transform, 
                tr=(0.0, 0.0), 
                num_people=self.num_people["test"]
            )


    def train_dataloader(self):
        dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size[Stage.TRAIN],
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )
        return dataloader

    def val_dataloader(self):
        dataloader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size[Stage.VAL],
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )
        return dataloader

    def test_dataloader(self):
        dataloader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size[Stage.TEST],
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )
        return dataloader

    def predict_dataloader(self):
        dataloader = DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size[Stage.PREDICT],
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )
        return dataloader