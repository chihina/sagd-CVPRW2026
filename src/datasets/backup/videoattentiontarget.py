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
from src.utils import square_bbox, Stage, generate_gaze_heatmap

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
# from src.utils import Stage, expand_bbox, generate_gaze_heatmap, generate_head_mask

TRAIN_SHOWS = [
    "Sherlock",
    "Hearing",
    "Modern_Family",
    "Cheers",
    "Star_Wars",
    "Veep",
    "BTS_at_Jimmy_Fallon",
    "Coveted",
    "Breaking_Bad",
    "Sound_of_Music",
    "Tartuffe",
    "Suits",
    "Driving_Miss_Daisy",
    "Crazy_Rich_Asian",
    "Keeping_Up_With_the_Kardashians",
    "Interview_at_the_Oscars",
    "Interview_with_Bill_Gates",
    "Arrested_Development",
    "A_Play_With_Words",
    "How_I_Met_Your_Mother",
    "Jersey_Shore",
    "My_Dinner_with_Andre",
    "Conan",
    "Band_of_Brothers",
    "The_View",
    "Seinfeld",
    "Grey's_Anatomy",
    "UFC_Octagon_Interview",
    "The_Ellen_Show",
    "Secret",
    "Friends",
    "Gone_with_the_Wind",
    "Three_Idiots",
    "All_in_the_Family",
    "Big_Bang_Theory",
    "Silicon_Valley",
    "Give_Me_One_Reason",
]
VAL_SHOWS = ["Orange_is_the_New_Black", "Before_Sunrise", "Project_Runway"]
TEST_SHOWS = [
    "CBS_This_Morning",
    "Downton_Abby",
    "Hell's_Kitchen",
    "I_Wanna_Marry_Harry",
    "It's_Always_Sunny_in_Philadelphia",
    "Jamie_Oliver",
    "MLB_Interview",
    "Survivor",
    "Titanic",
    "West_World",
]


class VideoAttentionTargetDataset(Dataset):
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
        self.annotations, self.paths = self.load_annotations()
               
        # load speaking status file
        self.df_speaking = self.load_annotations_speaker()
        self.df_speaking = self.df_speaking.groupby('path')
        
    
    def load_annotations_speaker(self):
        annotation_files = glob(os.path.join('/idiap/temp/agupta/data/attention/videoatttarget/speaker/', f"*/*.csv"))

        li = []
        for file in annotation_files:
            show, clip = file.split("/")[-2:]
            clip = clip.split('.')[0]
            df = pd.read_csv(file)

            # add column for path
            frame_names = glob(os.path.join(self.root, 'images', show, clip, '*.jpg'))
            frame_names.sort()
            paths = [os.path.join(show, clip, frame_names[int(f) - 1].split('/')[-1]) for f in df['frame'].values]
            df['path'] = paths
            
            df["split"] = "train"
            if show in VAL_SHOWS:
                df["split"] = "val"
            elif show in TEST_SHOWS:
                df["split"] = "test"

            li.append(df)
        annotations = pd.concat(li, axis=0, ignore_index=True)

        # Filter Annotations based on Split
        annotations = annotations[annotations["split"] == self.split].reset_index(
            drop=True
        )

        return annotations
    

    def load_annotations(self):
        annotation_files = glob(os.path.join(self.root, f"annotations/*/*/*/*.txt"))

        column_names = [
            "path",
            "head_xmin",
            "head_ymin",
            "head_xmax",
            "head_ymax",
            "gaze_x",
            "gaze_y",
        ]
        li = []
        for file in annotation_files:
            show, clip, fname = file.split("/")[-3:]
            df = pd.read_csv(file, names=column_names, sep=",")

            df["path"] = df["path"].apply(
                lambda img_name: os.path.join(show, clip, img_name)
            )
            df["pid"] = int(os.path.splitext(fname)[0][1:])  # "s02.txt" >> 2
            df["inout"] = (df["gaze_x"] != -1).astype(int)

            df["split"] = "train"
            if show in VAL_SHOWS:
                df["split"] = "val"
            elif show in TEST_SHOWS:
                df["split"] = "test"

            li.append(df)
        annotations = pd.concat(li, axis=0, ignore_index=True)
        # Filter Annotations based on Split
        annotations = annotations[annotations["split"] == self.split].reset_index(
            drop=True
        )
        
        # load looking at head GT file
        df_looking_head = pd.read_csv('/idiap/temp/agupta/data/attention/videoatttarget/gt_looking_head_target_'+self.split+'.csv')

        # join annotations and looking at head frames
        annotations = annotations.merge(df_looking_head, on=['path', 'pid'], how='left')
        
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

        # Load Image
        image = Image.open(os.path.join(self.root, "images", path)).convert("RGB")
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
        
        # Get speaking status and detected head bboxes
        if path in self.df_speaking.groups.keys():
            df_speaking_frame = self.df_speaking.get_group(path)
            det_head_bboxes = np.stack([df_speaking_frame['xmin'].values*img_w, df_speaking_frame['ymin'].values*img_h, df_speaking_frame['xmax'].values*img_w, df_speaking_frame['ymax'].values*img_h], axis=1)
            det_head_bboxes = torch.from_numpy(det_head_bboxes.astype(np.float32))
            speaking_det = torch.from_numpy(df_speaking_frame['score'].values.astype(np.float32))
            # Shuffle detected people
            if self.split == "train":
                rand_indices = torch.randperm(det_head_bboxes.size(0))
                det_head_bboxes = det_head_bboxes[rand_indices]
                speaking_det = speaking_det[rand_indices]
        else:
            det_head_bboxes = []
            speaking_scores = torch.zeros(len(head_bboxes), dtype=torch.float) - 1    # for annotated heads

        # Process detected head bboxes
        if len(det_head_bboxes) > 0:
            # merge annotated head bboxes
            ious = box_iou(det_head_bboxes, head_bboxes)
            ious_bbox, index_ious = torch.max(ious, axis=1)
            index_bbox_keep = ious_bbox < 0.5
            det_head_bboxes = det_head_bboxes[index_bbox_keep]
            head_bboxes = torch.concat([head_bboxes, det_head_bboxes], axis=0)
            
            ious_spk, index_ious = torch.max(ious, axis=0)
            index_spk_keep = ious_spk >= 0.5
            speaking_ann = speaking_det[index_ious] * index_spk_keep + (1-index_spk_keep.int())*-1
            speaking_det = speaking_det[index_bbox_keep]
            speaking_scores = torch.cat([speaking_ann, speaking_det])
            # concat -1 for gaze and inout annotations
            gaze_pts = torch.cat([gaze_pts, torch.zeros((len(det_head_bboxes), 2))-1])
            inout = torch.cat([inout, torch.zeros(len(det_head_bboxes), dtype=torch.float32)-1])
        
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
        speaking_scores = speaking_scores[:num_keep]
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
                "head_bboxes": head_bboxes,
                "inout": inout,
                "gaze_pts": gaze_pts,
                "num_valid_people": num_valid_heads,
                "img_size": torch.tensor((img_w, img_h), dtype=torch.long),
                "path": path,
                "dataset": 'vat'
        }
        
        # Transform
        if self.transform:
            sample = self.transform(sample)
            
        head_bboxes = sample['head_bboxes']
        gaze_pts = sample['gaze_pts']
        heads = sample['heads']

        # Pad missing people (ie. heads, head_bboxes, gaze_pt and coatt)
        if num_missing_heads > 0:
            head_bboxes = torch.cat([torch.zeros((num_missing_heads, 4), dtype=torch.float32), head_bboxes])
            heads = torch.cat([torch.zeros((num_missing_heads, 3, 224, 224), dtype=torch.float32), heads])
            gaze_pts = torch.cat([torch.zeros((num_missing_heads, 2), dtype=torch.float32)-1, gaze_pts])
            inout = torch.cat([torch.zeros((num_missing_heads, ), dtype=torch.float32)-1, inout])
            lah_ids[lah_ids>=0] += num_missing_heads
            lah_ids = torch.cat([torch.zeros((num_missing_heads, ), dtype=torch.long)-3, lah_ids])
            speaking_scores = torch.cat([torch.zeros((num_missing_heads, ), dtype=torch.long)-1, speaking_scores])
            
        # compute gaze vectors
        head_centers = torch.hstack([
            (head_bboxes[:, [0]] + head_bboxes[:, [2]]) / 2, 
            (head_bboxes[:, [1]] + head_bboxes[:, [3]]) / 2
        ])    
        
        # generate gaze heatmaps
        sample["gaze_heatmaps"] = generate_gaze_heatmap(gaze_pts, sigma=self.heatmap_sigma, size=self.heatmap_size)

        laeo_ids = torch.zeros(len(heads), dtype=torch.long)
        coatt_ids = torch.zeros(len(heads), dtype=torch.long)
        
        sample["gaze_vecs"] = F.normalize(gaze_pts - head_centers, p=2, dim=-1)
        sample['lah_ids'] = lah_ids
        sample['laeo_ids'] = laeo_ids
        sample['coatt_ids'] = coatt_ids
        sample['head_bboxes'] = head_bboxes
        sample['gaze_pts'] = gaze_pts
        sample['heads'] = heads
        sample['inout'] = inout
        sample['speaking'] = speaking_scores

        return sample

    def __len__(self):
        return len(self.paths)
        
    
class VideoAttentionTargetDataModule(pl.LightningDataModule):
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
                    RandomCropSafeGaze(aspect=1.33, p=1.0),
                    RandomHorizontalFlip(p=0.5),
                    ColorJitter(
                        brightness=(0.5, 1.5),
                        contrast=(0.5, 1.5),
                        saturation=(0.0, 1.5),
                        hue=None,
                        p=0.8,
                    ),
                    Resize(img_size=(224, 304), head_size=(224, 224)),
                    ToTensor(),
                    Normalize(
                        img_mean=[0.31072, 0.25703, 0.24182],
                        img_std=[0.26028, 0.24050, 0.23851]
                    ),
                ]
            )
            self.train_dataset = VideoAttentionTargetDataset(
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
                    Resize(img_size=(224, 304), head_size=(224, 224)),
                    ToTensor(),
                    Normalize(
                        img_mean=[0.31072, 0.25703, 0.24182],
                        img_std=[0.26028, 0.24050, 0.23851]
                    ),
                ]
            )
            self.val_dataset = VideoAttentionTargetDataset(
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
                    Resize(img_size=(224, 304), head_size=(224, 224)),
                    ToTensor(),
                    Normalize(
                        img_mean=[0.31072, 0.25703, 0.24182],
                        img_std=[0.26028, 0.24050, 0.23851]
                    ),
                ]
            )
            self.val_dataset = VideoAttentionTargetDataset(
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
                    Resize(img_size=224, head_size=(224, 224)),
                    ToTensor(),
                    Normalize(
                        img_mean=[0.31072, 0.25703, 0.24182],
                        img_std=[0.26028, 0.24050, 0.23851]
                    ),
                ]
            )
            self.test_dataset = VideoAttentionTargetDataset(
                root=self.root, 
                split="test", 
                stride=3,
                transform=test_transform, 
                tr=(0.0, 0.0), 
                heatmap_size=self.heatmap_size, 
                heatmap_sigma=self.heatmap_sigma,
                num_people=self.num_people['test']
            )
            

        elif stage == "predict":
            predict_transform = Compose(
                [
                    Resize(img_size=224, head_size=(224, 224)),
                    ToTensor(),
                    Normalize(
                        img_mean=[0.31072, 0.25703, 0.24182],
                        img_std=[0.26028, 0.24050, 0.23851]
                    ),
                ]
            )
            self.predict_dataset = VideoAttentionTargetDataset(
                root=self.root, 
                split="test",
                stride=1,
                transform=predict_transform, 
                tr=(0.0, 0.0), 
                heatmap_size=self.heatmap_size, 
                heatmap_sigma=self.heatmap_sigma,
                num_people=self.num_people['test']
            )


    def train_dataloader(self):
        dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size[Stage.TRAIN],
            shuffle=True,
            num_workers=8,
            pin_memory=False,
        )
        return dataloader

    def val_dataloader(self):
        dataloader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size[Stage.VAL],
            shuffle=False,
            num_workers=4,
            pin_memory=False,
        )
        return dataloader

    def test_dataloader(self):
        dataloader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size[Stage.TEST],
            shuffle=True,
            num_workers=4,
            pin_memory=False,
        )
        return dataloader

    def predict_dataloader(self):
        dataloader = DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size[Stage.PREDICT],
            shuffle=False,
            num_workers=8,
            pin_memory=False,
        )
        return dataloader