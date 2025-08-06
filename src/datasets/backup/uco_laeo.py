# IMPORTS
import os
from PIL import Image
from glob import glob
from typing import Tuple, Union

import numpy as np
import pandas as pd
import lightning.pytorch as pl

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.ops import box_iou

from src.utils import Stage, square_bbox, generate_gaze_heatmap
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

IMG_MEAN = [0.44232, 0.40506, 0.36457]
IMG_STD = [0.28674, 0.27776, 0.27995]


# ============================================================================================================ #
#                                               VIDEOLAEO DATASET                                             #
# ============================================================================================================ #
class VideoLAEODataset(Dataset):
    def __init__(
        self,
        root: str,
        split: str = "train",
        stride: int = 1,
        transform: Union[Compose, None] = None,
        tr: Tuple[float, float] = (-0.1, 0.1),
        heatmap_sigma=3,
        heatmap_size=64,
        num_people: Union[int, str] = 5
    ):
        super().__init__()
        self.root = root
        self.split = "validate" if split == "val" else split
        self.stride = stride
        self.jitter_bbox = RandomHeadBboxJitter(p=1.0, tr=tr)
        self.transform = transform
        self.heatmap_sigma = heatmap_sigma
        self.heatmap_size = heatmap_size
        self.num_people = num_people
        self.annotations, self.paths, self.path2indices = self.load_annotations()
        
        # load speaking status file
        self.df_speaking = self.load_annotations_speaker()
        self.df_speaking = self.df_speaking.groupby('path')
        
    def load_annotations_speaker(self):
        annotation_files = glob(os.path.join('/idiap/temp/agupta/data/UCO-LAEO/speaker', f"*.csv"))

        li = []
        for file in annotation_files:
            clip = file.split('/')[-1]
            clip = clip.split('.')[0]
            df = pd.read_csv(file)
            
            # add column for path
            frame_names = glob(os.path.join(self.root, 'images_Idiap', 'frames', clip, '*.jpg'))
            frame_names.sort()
            paths = [os.path.join('frames', clip, frame_names[int(f) - 1].split('/')[-1]) for f in df['frame'].values]
            df['path'] = paths

            li.append(df)
        annotations = pd.concat(li, axis=0, ignore_index=True)

        return annotations

    def load_annotations(self):
        # Change to LAEO annotations
        # annotation_files = sorted(glob(f"/idiap/temp/stafasca/data/VideoCoAtt/annotations/{self.split}/*.csv"))
        annotation_files = sorted(glob(f"/idiap/home/nchutisilp/laeo_datasets/{self.split}/*.csv"))
        # annotation_files = sorted(glob(f"C:/Users/Namka/Documents/EPFL/Semester 4/Semester Project/mangekyo-sharingan-master/laeo_datasets/{self.split}/*.csv"))

        li = []
        for file in annotation_files:
            df = pd.read_csv(file, sep=",")
            li.append(df)
        annotations = pd.concat(li, axis=0, ignore_index=True)
        path2indices = annotations.groupby("path").indices
        paths = annotations.path.unique()
        
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
        image = Image.open(os.path.join(self.root, "images_Idiap", path)).convert("RGB")
        img_w, img_h = image.size
        
        # Get person head bounding boxes
        head_bboxes = img_annotations[["head_xmin", "head_ymin", "head_xmax", "head_ymax"]]
        head_bboxes = torch.from_numpy(head_bboxes.values.astype(np.float32))
        # keep valid head bboxes
        head_bboxes = [hb for hb in head_bboxes if hb[0]!=-1]
        if len(head_bboxes)>0:
            head_bboxes = torch.stack(head_bboxes)
        else:
            head_bboxes = torch.tensor(head_bboxes)
        
        # jitter head bboxes
        if self.split == "train":
            head_bboxes = self.jitter_bbox(head_bboxes, img_w, img_h) 
            
        # Square head bboxes (can have negative values)
        num_heads = len(head_bboxes)
        if num_heads>0:
            head_bboxes = square_bbox(head_bboxes, img_w, img_h)

        # Extract Heads
        heads = []
        for head_bbox in head_bboxes:
            heads.append(image.crop(head_bbox.int().tolist()))

        # Create (Normalized) Gaze Points
        gaze_pts = torch.from_numpy(img_annotations[["gaze_x", "gaze_y"]].values.astype(np.float32))
        gaze_pts[gaze_pts[:, 0] != -1.] /= torch.tensor([img_w, img_h])
        
        # Get Coatt id and bounding boxes
        laeo_ids = torch.from_numpy(img_annotations["laeo_id"].values.astype(int))
        
        # Get detected head bboxes and speaking scores
        det_head_bboxes = []
        speaking_scores = torch.zeros(len(head_bboxes), dtype=torch.float) - 1    # for annotated heads
        if path in self.df_speaking.groups.keys():
            df_speaking_frame = self.df_speaking.get_group(path)
            det_head_bboxes = np.stack([df_speaking_frame['xmin'].values*img_w, df_speaking_frame['ymin'].values*img_h, df_speaking_frame['xmax'].values*img_w, df_speaking_frame['ymax'].values*img_h], axis=1)
            det_head_bboxes = torch.from_numpy(det_head_bboxes.astype(np.float32))
            speaking_det = torch.from_numpy(df_speaking_frame['score'].values.astype(np.float32))                      

        # Match annotated and detected head bboxes
        if len(det_head_bboxes) > 0 and len(head_bboxes)>0:
            # merge annotated head bboxes
            ious = box_iou(det_head_bboxes, head_bboxes)
            ious_spk, index_ious = torch.max(ious, axis=0)
            index_spk_keep = ious_spk >= 0.5
            speaking_scores = speaking_det[index_ious] * index_spk_keep + (1-index_spk_keep.int())*-1

        # Randomly select a subset of people during training
        if self.split == "train":
            # Select upto `num_people` people
            num_keep = num_heads
            if self.num_people!='all':
                if num_heads>1:
                    num_keep = np.random.randint(2, min(num_heads, self.num_people)+1)
            rand_indices = torch.randperm(len(head_bboxes))
            head_bboxes = head_bboxes[rand_indices][:num_keep]  # shuffle rows, keep first `num_people` head bboxes
            heads = [heads[i] for i in rand_indices[:num_keep]]
            gaze_pts = gaze_pts[rand_indices][:num_keep]
            laeo_ids = laeo_ids[rand_indices][:num_keep]
            speaking_scores = speaking_scores[rand_indices][:num_keep]
        num_valid_heads = len(heads)
        num_missing_heads = max(self.num_people + 1 - num_valid_heads, 1) if self.num_people != "all" else 1    # pad at least one person

        # Normalize Head Bboxes
        if num_valid_heads>0:
            head_bboxes /= torch.tensor([img_w, img_h, img_w, img_h], dtype=float)

        # get inout values
        inout = torch.zeros(len(heads), dtype=torch.float32) - 1
        inout[laeo_ids>0] = 1
        
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
                "dataset": 'laeo'
        }
        
        # Transform
        if self.transform:
            sample = self.transform(sample)
            head_bboxes = sample['head_bboxes']
            gaze_pts = sample['gaze_pts']
            heads = sample['heads']
        if len(heads)==0:
            heads = torch.tensor(heads)

        # Pad missing people (ie. heads, head_bboxes, gaze_pt and coatt)
        if num_missing_heads > 0:
            head_bboxes = torch.cat([torch.zeros((num_missing_heads, 4), dtype=torch.float32), head_bboxes])
            heads = torch.cat([torch.zeros((num_missing_heads, 3, 224, 224), dtype=torch.float32), heads])
            gaze_pts = torch.cat([torch.zeros((num_missing_heads, 2), dtype=torch.float32)-1, gaze_pts])
            inout = torch.cat([torch.zeros((num_missing_heads, ), dtype=torch.float32)-1, inout])
            laeo_ids = torch.cat([torch.zeros((num_missing_heads, ), dtype=torch.long), laeo_ids])
            speaking_scores = torch.cat([torch.zeros((num_missing_heads, ), dtype=torch.float32)-1, speaking_scores])
            
        # compute gaze vectors
        head_centers = torch.hstack([
            (head_bboxes[:, [0]] + head_bboxes[:, [2]]) / 2, 
            (head_bboxes[:, [1]] + head_bboxes[:, [3]]) / 2
        ])    
        
        # generate gaze heatmaps
        sample["gaze_heatmaps"] = generate_gaze_heatmap(gaze_pts, sigma=self.heatmap_sigma, size=self.heatmap_size)

        lah_ids = torch.zeros(len(heads), dtype=torch.long) - 3
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
        return self.length
    
    

# ============================================================================================================ #
#                                              VIDEOLAEO DATA MODULE                                          #
# ============================================================================================================ #
class VideoLAEODataModule(pl.LightningDataModule):
    def __init__(
        self,
        root: str,
        batch_size: Union[int, dict] = 32,
        num_people: int = 5
    ):  
        
        super().__init__()
        self.root = root
        self.num_people = num_people
        if isinstance(num_people, dict):
            print('num_people', num_people)

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
            self.train_dataset = VideoLAEODataset(
                root=self.root, 
                split="train", 
                stride=3,
                transform=train_transform, 
                tr=(-0.1, 0.1), 
                num_people=self.num_people["train"]
            )

            val_transform = Compose(
                [
                    Resize(img_size=(224, 224), head_size=(224, 224)),
                    ToTensor(),
                    Normalize(
                        img_mean=IMG_MEAN,
                        img_std=IMG_STD
                    ),
                ]
            )
            self.val_dataset = VideoLAEODataset(
                root=self.root, 
                split="val", 
                stride=3,
                transform=val_transform, 
                tr=(0.0, 0.0), 
                num_people=self.num_people["val"]
            )

        elif stage == "validate":
            val_transform = Compose(
                [
                    Resize(img_size=(224, 224), head_size=(224, 224)),
                    ToTensor(),
                    Normalize(
                        img_mean=IMG_MEAN,
                        img_std=IMG_STD
                    ),
                ]
            )
            self.val_dataset = VideoLAEODataset(
                root=self.root, 
                split="val",
                stride=3,
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
            self.test_dataset = VideoLAEODataset(
                root=self.root, 
                split="test", 
                stride=3,
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
            self.predict_dataset = VideoLAEODataset(
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
            num_workers=8,
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
