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
from src.utils import Stage, expand_bbox, generate_gaze_heatmap, square_bbox


class VacationDataset(Dataset):
    def __init__(
        self,
        root: str,
        split: str = "train",
        stride: int = 3,
        transform: Union[Compose, None] = None,
        tr=(-0.1, 0.1),
        heatmap_sigma=3,
        heatmap_size=64,
        num_people=5
    ):
        super().__init__()
        self.root = root
        self.split = split
        self.stride = stride
        self.jitter_bbox = RandomHeadBboxJitter(p=1.0, tr=tr)
        self.transform = transform
        self.heatmap_sigma = heatmap_sigma
        self.heatmap_size = heatmap_size
        self.num_people = num_people
        # load annotations
        self.annotations, self.paths = self.load_annotations()
        

    def load_annotations(self):
        annotation_path = os.path.join(self.root, 'processed_csvs', f'{self.split}.csv')
        annotations = pd.read_csv(annotation_path)

        # group by path
        annotations = annotations.groupby('path')
        paths = list(annotations.groups.keys())
        paths = np.array(paths)
        
        if self.stride > 1:
            index_keep = np.arange(len(paths), step=self.stride)
            paths = paths[index_keep]

        return annotations, paths

    
    def __getitem__(self, index):
        # Load annotations
        path = self.paths[index]
        img_annotations = self.annotations.get_group(path)
        inout = torch.from_numpy(img_annotations["inout"].values.astype(np.float32))
        gaze_pts = torch.from_numpy(img_annotations[["gaze_x", "gaze_y"]].values.astype(np.float32))
        laeo_ids = torch.from_numpy(img_annotations["laeo_id"].values)
        coatt_ids = torch.from_numpy(img_annotations["coatt_id"].values)

        # Load Image
        image = Image.open(os.path.join(self.root, path)).convert("RGB")
        img_w, img_h = image.size

        # Get annotated person head bounding boxes
        head_bboxes = img_annotations[["head_xmin", "head_ymin", "head_xmax", "head_ymax"]]
        head_bboxes = torch.from_numpy(head_bboxes.values.astype(np.float32))
        
        # Shuffle annotated people
        if self.split == "train":
            rand_indices = torch.randperm(head_bboxes.size(0))
            head_bboxes = head_bboxes[rand_indices]
            gaze_pts = gaze_pts[rand_indices]
            inout = inout[rand_indices]
            laeo_ids = laeo_ids[rand_indices]
            coatt_ids = coatt_ids[rand_indices]
        
        # jitter head bboxes
        if self.split == "train":
            head_bboxes = self.jitter_bbox(head_bboxes, img_w, img_h) 
            
        # Square head bboxes (can have negative values)
        head_bboxes = square_bbox(head_bboxes, img_w, img_h)
        num_heads = len(head_bboxes)

        # Extract Heads
        heads = []
        for head_bbox in head_bboxes:
            heads.append(image.crop(head_bbox.int().tolist()))
            
        # Select upto `num_people` people
        num_keep = num_heads
        if self.num_people!='all':
            if num_heads>1:
                num_keep = np.random.randint(2, min(num_heads, self.num_people)+1)
        head_bboxes = head_bboxes[:num_keep]
        heads = heads[:num_keep]
        gaze_pts = gaze_pts[:num_keep]
        inout = inout[:num_keep]
        laeo_ids = laeo_ids[:num_keep]
        coatt_ids = coatt_ids[:num_keep]
        num_valid_heads = len(heads)
        num_missing_heads = max(self.num_people + 1 - num_valid_heads, 1) if self.num_people != "all" else 1    # pad at least one person
                                     
        # Get lah id
        lah_ids = torch.zeros(len(heads), dtype=torch.long) - 5
        for i in range(len(heads)):
            gaze_pt = gaze_pts[i]
            io = inout[i]
            if io==0:
                lah_ids[i] = -1
            elif io==1:
                # check if gaze point is inside any head bbox
                inside1 = (head_bboxes[:, :2] < gaze_pt).int().prod(1)
                inside2 = (head_bboxes[:, 2:] > gaze_pt).int().prod(1)
                lah_id = np.where(inside1 * inside2)[0]
                if len(lah_id)>=1:
                    lah_id = lah_id[:1].item()
                    if lah_id==i:
                        lah_id = -1
                    lah_ids[i] = lah_id
                else:
                    lah_ids[i] = -1

        # Create (Normalized) Gaze Points
        gaze_pts[gaze_pts[:, 0] != -1.] /= torch.tensor([img_w, img_h])
        
        # Normalize Head Bboxes
        head_bboxes /= torch.tensor([img_w, img_h, img_w, img_h], dtype=float)  
                    
        # Build Sample
        sample = {
                "image": image,
                "heads": heads,
                "pcd": torch.zeros((3,img_h,img_w), dtype=torch.float32),
                "head_bboxes": head_bboxes,
                "inout": inout,
                "gaze_pts": gaze_pts,
                "num_valid_people": torch.tensor(num_valid_heads).unsqueeze(0),
                "img_size": torch.tensor((img_w, img_h), dtype=torch.long),
                "path": path,
                "dataset": 'vacation'
        }
        
        # Transform
        if self.transform:
            sample = self.transform(sample)
            
        head_bboxes = sample['head_bboxes']
        gaze_pts = sample['gaze_pts']
        heads = sample['heads']
        image = sample['image']

        # Pad missing people (ie. heads, head_bboxes, gaze_pt and coatt)
        if num_missing_heads > 0:
            head_bboxes = torch.cat([torch.zeros((num_missing_heads, 4), dtype=torch.float32), head_bboxes])
            heads = torch.cat([torch.zeros((num_missing_heads, 3, 224, 224), dtype=torch.float32), heads])
            gaze_pts = torch.cat([torch.zeros((num_missing_heads, 2), dtype=torch.float32)-1, gaze_pts])
            inout = torch.cat([torch.zeros((num_missing_heads, ), dtype=torch.float32)-1, inout])
            lah_ids[lah_ids>=0] += num_missing_heads
            lah_ids = torch.cat([torch.zeros((num_missing_heads, ), dtype=torch.long)-3, lah_ids])
            laeo_ids = torch.cat([torch.zeros((num_missing_heads, ), dtype=torch.long), laeo_ids])
            coatt_ids = torch.cat([torch.zeros((num_missing_heads, ), dtype=torch.long), coatt_ids])
            
        # compute gaze vectors
        head_centers = torch.hstack([
            (head_bboxes[:, [0]] + head_bboxes[:, [2]]) / 2, 
            (head_bboxes[:, [1]] + head_bboxes[:, [3]]) / 2
        ])    
        
        # generate gaze heatmaps
        sample['image'] = image.unsqueeze(0)
        sample["gaze_heatmaps"] = generate_gaze_heatmap(gaze_pts, sigma=self.heatmap_sigma, size=self.heatmap_size)
        sample["gaze_vecs"] = F.normalize(gaze_pts - head_centers, p=2, dim=-1).unsqueeze(0)    # include dimension for time
        sample['lah_ids'] = lah_ids.unsqueeze(0)
        sample['laeo_ids'] = laeo_ids.unsqueeze(0)
        sample['coatt_ids'] = coatt_ids.unsqueeze(0)
        sample['head_bboxes'] = head_bboxes.unsqueeze(0)
        sample['gaze_pts'] = gaze_pts.unsqueeze(0)
        sample['heads'] = heads.unsqueeze(0)
        sample['inout'] = inout.unsqueeze(0)
        sample['is_child'] = torch.zeros(len(lah_ids)).unsqueeze(0)-1
        sample['speaking'] = torch.zeros(len(lah_ids)).unsqueeze(0)-1

        return sample

    def __len__(self):
        return len(self.paths)   
    
    
class VacationDataModule(pl.LightningDataModule):
    def __init__(
        self,
        root: str,
        batch_size: Union[int, dict] = 32,
        predict_input_file: Union[str, None] = None,
        predict_annotation_file: Union[str, None] = None,
        heatmap_sigma: int = 3,
        heatmap_size = 64,
        num_people: int = 5
    ):  
        
        super().__init__()
        self.root = root
        self.heatmap_sigma = heatmap_sigma
        self.heatmap_size = heatmap_size
        self.num_people = num_people
        self.batch_size = (
            {stage: batch_size for stage in Stage}
            if isinstance(batch_size, int)
            else batch_size
        )
        self.predict_input_file = predict_input_file
        self.predict_annotation_file = predict_annotation_file


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
                    # Resize(img_size=(224, 224), head_size=(224, 224)),
                    Resize(img_size=(self.cfg.data.image_size, self.cfg.data.image_size), head_size=(self.cfg.data.image_size, self.cfg.data.image_size)),
                    ToTensor(),
                    Normalize(
                        img_mean=[0.31072, 0.25703, 0.24182],
                        img_std=[0.26028, 0.24050, 0.23851]
                    ),
                ]
            )
            self.train_dataset = VacationDataset(
                root=self.root, 
                split="train", 
                stride=3,
                transform=train_transform, 
                tr=(-0.1, 0.1), 
                heatmap_size=self.heatmap_size, 
                heatmap_sigma=self.heatmap_sigma,
                num_people=self.num_people['train']
            )

            val_transform = Compose(
                [
                    # Resize(img_size=(224, 224), head_size=(224, 224)),
                    Resize(img_size=(self.cfg.data.image_size, self.cfg.data.image_size), head_size=(self.cfg.data.image_size, self.cfg.data.image_size)),
                    ToTensor(),
                    Normalize(
                        img_mean=[0.31072, 0.25703, 0.24182],
                        img_std=[0.26028, 0.24050, 0.23851]
                    ),
                ]
            )
            self.val_dataset = VacationDataset(
                root=self.root, 
                split="val", 
                stride=3,
                transform=val_transform, 
                tr=(0.0, 0.0), 
                heatmap_size=self.heatmap_size, 
                heatmap_sigma=self.heatmap_sigma,
                num_people=self.num_people['val']
            )

        elif stage == "validate":
            val_transform = Compose(
                [
                    # Resize(img_size=(224, 224), head_size=(224, 224)),
                    Resize(img_size=(self.cfg.data.image_size, self.cfg.data.image_size), head_size=(self.cfg.data.image_size, self.cfg.data.image_size)),
                    ToTensor(),
                    Normalize(
                        img_mean=[0.31072, 0.25703, 0.24182],
                        img_std=[0.26028, 0.24050, 0.23851]
                    ),
                ]
            )
            self.val_dataset = VacationDataset(
                root=self.root, 
                split="val",
                stride=3,
                transform=val_transform, 
                tr=(0.0, 0.0), 
                heatmap_size=self.heatmap_size, 
                heatmap_sigma=self.heatmap_sigma,
                num_people=self.num_people['val']
            )

        elif stage == "test":
            test_transform = Compose(
                [
                    # Resize(img_size=(224, 224), head_size=(224, 224)),
                    Resize(img_size=(self.cfg.data.image_size, self.cfg.data.image_size), head_size=(self.cfg.data.image_size, self.cfg.data.image_size)),
                    ToTensor(),
                    Normalize(
                        img_mean=[0.31072, 0.25703, 0.24182],
                        img_std=[0.26028, 0.24050, 0.23851]
                    ),
                ]
            )
            self.test_dataset = VacationDataset(
                root=self.root, 
                split="test", 
                stride=3,
                transform=test_transform, 
                tr=(0.0, 0.0), 
                heatmap_size=self.heatmap_size, 
                heatmap_sigma=self.heatmap_sigma,
                num_people='all'
            )
            

        elif stage == "predict":
            predict_transform = Compose(
                [
                    # Resize(img_size=(224, 224), head_size=(224, 224)),
                    Resize(img_size=(self.cfg.data.image_size, self.cfg.data.image_size), head_size=(self.cfg.data.image_size, self.cfg.data.image_size)),
                    ToTensor(),
                    Normalize(
                        img_mean=[0.31072, 0.25703, 0.24182],
                        img_std=[0.26028, 0.24050, 0.23851]
                    ),
                ]
            )
            self.predict_dataset = VacationDataset(
                root=self.root, 
                split="test",
                stride=1,
                transform=predict_transform, 
                tr=(0.0, 0.0), 
                heatmap_size=self.heatmap_size, 
                heatmap_sigma=self.heatmap_sigma,
                num_people=self.num_people
            )


    def train_dataloader(self):
        dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size[Stage.TRAIN],
            shuffle=True,
            num_workers=8,
            pin_memory=True,
        )
        return dataloader

    def val_dataloader(self):
        dataloader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size[Stage.VAL],
            shuffle=False,
            num_workers=8,
            pin_memory=True,
        )
        return dataloader

    def test_dataloader(self):
        dataloader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size[Stage.TEST],
            shuffle=True,
            num_workers=8,
            pin_memory=True,
        )
        return dataloader

    def predict_dataloader(self):
        dataloader = DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size[Stage.PREDICT],
            shuffle=False,
            num_workers=8,
            pin_memory=True,
        )
        return dataloader