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
from src.utils import square_bbox, Stage, generate_gaze_heatmap, lah2laeo, lah2coatt, generate_mask, pair

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


class VideoAttentionTargetDataset_temporal(Dataset):
    def __init__(
        self,
        root: str,
        split: str = "train",
        stride: int = 3,
        transform: Union[Compose, None] = None,
        tr=(-0.1, 0.1),
        image_size=(224,224),
        heatmap_sigma=3,
        heatmap_size=64,
        num_people=5,
        temporal_stride = 3,
        temporal_context = 2,
        aspect=False
    ):
        super().__init__()
        self.root = root
        self.split = split
        self.stride = stride
        self.jitter_bbox = RandomHeadBboxJitter(p=1.0, tr=tr)
        self.transform = transform
        self.image_size = image_size
        self.heatmap_sigma = heatmap_sigma
        self.heatmap_size = heatmap_size
        self.num_people = num_people
        self.temporal_stride = temporal_stride
        self.temporal_context = temporal_context
        self.pid_offset = 1000
        self.aspect = aspect
               
        # load annotations
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
            df["person_id"] = int(os.path.splitext(fname)[0][1:])  # "s02.txt" >> 2
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
                
        # group by path
        annotations = annotations.groupby('path')
        paths = list(annotations.groups.keys())
        paths = np.array(paths)
        
        if self.stride > 1:
            index_keep = np.arange(len(paths), step=self.stride)
            paths = paths[index_keep]

        return annotations, paths

    
    def __getitem__(self, index):
        path = self.paths[index]
        clip = '/'.join(path.split('/')[:-1])
        frame = int(path.split('/')[-1][:-4])
        # jitter frame number during training
        if self.split=='train':
            if self.temporal_context==0:
                frame_shift = torch.randint(-(self.temporal_stride//2), self.temporal_stride//2, (1,)).item()
            else:
                frame_shift = torch.randint(-(self.temporal_context*self.temporal_stride-1), self.temporal_context*self.temporal_stride, (1,)).item()
            frame_tmp = frame + frame_shift
            path = os.path.join(clip,  str(frame_tmp).zfill(8)+'.jpg')
            while path not in self.annotations.groups.keys():
                if self.temporal_context==0:
                    frame_shift = torch.randint(-(self.temporal_stride//2), self.temporal_stride//2, (1,)).item()
                else:
                    frame_shift = torch.randint(-(self.temporal_context*self.temporal_stride-1), self.temporal_context*self.temporal_stride, (1,)).item()
                frame_tmp = frame + frame_shift
                path = os.path.join(clip,  str(frame_tmp).zfill(8)+'.jpg')
            frame = frame_tmp
        img_annotations = self.annotations.get_group(path)
        
        # get current frame num
        curr_frame_nb = frame

        # read current frame
        img_path = os.path.join(self.root, 'images', path)
        image = Image.open(img_path)  
        img_w, img_h = image.size
        if self.split=='test' and self.aspect:    # maintain aspect ratio
            dummy_sample = {}
            dummy_sample['image'] = image
            dummy_sample['heads'] = []
            dummy_sample['pcd'] = torch.zeros((3,img_h,img_w), dtype=torch.float32)
            dummy_sample = Resize(img_size=self.image_size[1], head_size=(224,224))(dummy_sample)
            self.image_size = dummy_sample['image'].size
        
        # get frame nums around current frame
        frame_nbs = np.arange(curr_frame_nb-(self.temporal_stride*self.temporal_context), curr_frame_nb+(self.temporal_stride*self.temporal_context)+1, self.temporal_stride)
        
        # get annotated person ids
        pids_ann = img_annotations['person_id'].values
        head_bboxes = img_annotations[["head_xmin", "head_ymin", "head_xmax", "head_ymax"]]
        head_bboxes = torch.from_numpy(head_bboxes.values.astype(np.float32))
        if self.split=='train':    # shuffle pids
            rand_indices = torch.randperm(head_bboxes.size(0))
            pids_ann = pids_ann[rand_indices]
        if len(pids_ann.shape)==0:
            pids_ann = np.expand_dims(pids_ann, 0)
        
        # get detected person ids
        pids_det = torch.tensor([])
        if path in self.df_speaking.groups.keys():
            df_speaking_frame = self.df_speaking.get_group(path)
            det_head_bboxes = np.stack([df_speaking_frame['xmin'].values*img_w, df_speaking_frame['ymin'].values*img_h, df_speaking_frame['xmax'].values*img_w, df_speaking_frame['ymax'].values*img_h], axis=1)
            det_head_bboxes = torch.from_numpy(det_head_bboxes.astype(np.float32))
            pids_det = df_speaking_frame['id'].values + self.pid_offset    # add offset to detected people ids
        
        # keep non-overlapping detected heads
        index_spk = torch.zeros(len(pids_ann)); index_spk_keep = torch.zeros(len(pids_ann))
        if len(pids_det) > 0:
            # merge annotated head bboxes
            ious = box_iou(det_head_bboxes, head_bboxes)
            ious_spk, index_spk = torch.max(ious, axis=0)
            index_spk = pids_det[index_spk]
            index_spk_keep = ious_spk >= 0.5
            if len(index_spk.shape)==0:
                index_spk = torch.tensor([index_spk])
                index_spk_keep = torch.tensor([index_spk_keep])
            
            ious_bbox, _ = torch.max(ious, axis=1)
            index_bbox_keep = ious_bbox < 0.3
            pids_det = pids_det[index_bbox_keep.numpy()]
            if len(pids_det.shape)==0:
                pids_det = np.expand_dims(pids_det, 0)
            elif self.split=='train' and len(pids_det)>1:    # shuffle pids
                rand_indices = torch.randperm(len(pids_det))
                pids_det = pids_det[rand_indices]
        
        # keep up to num_people ids
        person_ids = np.concatenate([pids_ann, pids_det])
        num_heads = len(person_ids)
        num_keep = num_heads
        if self.num_people!='all':
            batch_num_heads = self.num_people
            if num_heads>1:
                num_keep = np.random.randint(2, min(num_heads, self.num_people)+1)
        else:
            batch_num_heads = num_heads
        person_ids = person_ids[:num_keep]
        person_ids_ann = person_ids[person_ids<self.pid_offset]
        person_ids_det = person_ids[person_ids>=self.pid_offset]
        
        # randomly choose to apply the horizontal flip augmentation
        self.horizontal_flip = False
        if self.split=='train' and torch.rand(1) <= 0.5:
            self.horizontal_flip = RandomHorizontalFlip(p=1)

# define temporal sample
        t_sample = {
                "image": [],
                "heads": [],
                "head_centers": [],
                "head_masks": [],
                "head_bboxes": [],
                "inout": [],
                "gaze_pts": [],
                "gaze_vecs": [],
                "gaze_heatmaps": [],
                "lah_ids": [],
                "laeo_ids": [],
                "coatt_ids": [],
                "speaking": [],
                "is_child": [],
                "num_valid_people": [],
                "img_size": [],
                "path": [],
                "dataset": 'videoattentiontarget'
                }
        t_sample['pids'] = torch.cat([torch.zeros((batch_num_heads + 1 -len(person_ids), ))-1, torch.from_numpy(person_ids)])
        for frame_nb in frame_nbs:
            # check if frame exists
            path = os.path.join(clip,  str(frame_nb).zfill(8)+'.jpg')
            if not path in self.annotations.groups.keys():
                t_sample['image'].append(torch.zeros((3, self.image_size[1], self.image_size[0]), dtype=torch.float32))
                t_sample['heads'].append(torch.zeros((batch_num_heads+1, 3, 224, 224), dtype=torch.float32))
                t_sample['head_centers'].append(torch.zeros((batch_num_heads+1, 2), dtype=torch.float32))
                t_sample['head_masks'].append(torch.zeros((batch_num_heads+1, 1, self.image_size[1], self.image_size[0]), dtype=torch.float32))
                t_sample['head_bboxes'].append(torch.zeros((batch_num_heads+1, 4), dtype=torch.float32))
                t_sample['gaze_pts'].append(torch.zeros((batch_num_heads+1, 2), dtype=torch.float32)-1)
                t_sample['gaze_vecs'].append(torch.zeros((batch_num_heads+1, 2), dtype=torch.float32))
                t_sample['gaze_heatmaps'].append(torch.zeros((batch_num_heads+1, self.heatmap_size, self.heatmap_size), dtype=torch.float32))
                t_sample['inout'].append(torch.zeros((batch_num_heads+1), dtype=torch.float32)-1)
                t_sample['lah_ids'].append(torch.zeros((batch_num_heads+1), dtype=torch.long)-3)
                t_sample['laeo_ids'].append(torch.zeros((batch_num_heads+1), dtype=torch.long))
                t_sample['coatt_ids'].append(torch.zeros((batch_num_heads+1), dtype=torch.long))
                t_sample['speaking'].append(torch.zeros((batch_num_heads+1), dtype=torch.float32)-1)
                t_sample['is_child'].append(torch.zeros((batch_num_heads+1), dtype=torch.float32)-1)
                t_sample['num_valid_people'].append(torch.zeros(1, dtype=torch.long))
                t_sample['img_size'].append(torch.zeros(2, dtype=torch.long))
                t_sample['path'].append('')
            else:
                ###########################################
                # Get annotations
                ###########################################
                # Load image
                path = os.path.join(clip,  str(frame_nb).zfill(8)+'.jpg')
                img_path = os.path.join(self.root, 'images', path)
                image = Image.open(img_path)  
                img_w, img_h = image.size
                
                pcd = torch.zeros((3, img_h, img_w), dtype=torch.float32)
                
                # Get detected head bboxes
                if path in self.df_speaking.groups.keys():
                    df_speaking_frame = self.df_speaking.get_group(path)
                    det_head_bboxes = np.stack([df_speaking_frame['xmin'].values*img_w, df_speaking_frame['ymin'].values*img_h, df_speaking_frame['xmax'].values*img_w, df_speaking_frame['ymax'].values*img_h], axis=1)
                    det_head_bboxes = torch.from_numpy(det_head_bboxes.astype(np.float32))
                    pids_det = torch.from_numpy(df_speaking_frame['id'].values) + self.pid_offset
                    speaking_det = torch.from_numpy(df_speaking_frame['score'].values.astype(np.float32))
                else:
                    pids_det = torch.tensor([])
                    det_head_bboxes = []

                # Load annotations for selected person ids
                img_annotations = self.annotations.get_group(path)
                pids_ann = img_annotations["person_id"].values 
                head_bboxes = []; gaze_pts = []; inout = []; speaking_scores = []
                for pi, pid in enumerate(person_ids_ann):
                    pid_idx = np.where(pids_ann==pid)[0]
                    if len(pid_idx)==0:
                        head_bboxes.append(torch.zeros(4, dtype=torch.float32))
                        gaze_pts.append(torch.zeros(2, dtype=torch.float32)-1)
                        inout.append(-1)
                        speaking_scores.append(-1)
                    else:
                        img_ann = img_annotations.iloc[pid_idx]
                        head_bbox = img_ann[["head_xmin", "head_ymin", "head_xmax", "head_ymax"]]
                        head_bbox = torch.from_numpy(head_bbox.values.astype(np.float32)).squeeze() 
                        head_bboxes.append(head_bbox)
                        gaze_pt = img_ann[["gaze_x", "gaze_y"]]
                        gaze_pt = torch.from_numpy(gaze_pt.values.astype(np.float32)).squeeze() 
                        gaze_pts.append(gaze_pt)
                        io = img_ann['inout'].values.item()
                        inout.append(io)
                        if len(det_head_bboxes)>0:
                            pid_idx = torch.where(pids_det==index_spk[pi])[0]
                            if len(pid_idx)>0 and index_spk_keep[pi]:
                                speaking_scores.append(speaking_det[pid_idx])
                            else:
                                speaking_scores.append(-1)
                        else:
                            speaking_scores.append(-1)

                # Process detected head bboxes
                for pid in person_ids_det:
                    pid_idx = np.where(pids_det==pid)[0]
                    gaze_pts.append(torch.zeros(2, dtype=torch.float32)-1)
                    inout.append(-1)
                    if len(pid_idx)==0:
                        speaking_scores.append(-1)
                        head_bboxes.append(torch.zeros(4, dtype=torch.float32))
                    else:
                        speaking_scores.append(speaking_det[pid_idx])
                        head_bboxes.append(det_head_bboxes[pid_idx].squeeze())

                # stack annotations
                head_bboxes = torch.stack(head_bboxes)
                gaze_pts = torch.stack(gaze_pts)
                inout = torch.tensor(inout, dtype=torch.float)
                speaking_scores = torch.tensor(speaking_scores, dtype=torch.float)
                
                # jitter head bboxes
                if self.split == "train":
                    head_bboxes = self.jitter_bbox(head_bboxes, img_w, img_h) 

                # Square head bboxes (can have negative values)
                head_bboxes = square_bbox(head_bboxes, img_w, img_h)

                # Extract Heads
                heads = []
                for head_bbox in head_bboxes:
                    heads.append(image.crop(head_bbox.int().tolist()))
                num_valid_heads = len(heads)
#                 num_missing_heads = max(self.num_people - num_valid_heads, 0) if self.num_people != "all" else 0    
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
                        "pcd": pcd,
                        "heads": heads,
                        "head_bboxes": head_bboxes,
                        "inout": inout,
                        "gaze_pts": gaze_pts,
                        "num_valid_people": torch.tensor([num_valid_heads]),
                        "img_size": torch.tensor((img_w, img_h), dtype=torch.long),
                        "path": path
                }

                # Transform
                if self.transform:
                    sample = self.transform(sample)
                if self.horizontal_flip:
                    sample = self.horizontal_flip(sample)

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
                    speaking_scores = torch.cat([torch.zeros((num_missing_heads, ), dtype=torch.float32)-1, speaking_scores])

                # compute gaze vectors
                head_centers = torch.hstack([
                    (head_bboxes[:, [0]] + head_bboxes[:, [2]]) / 2, 
                    (head_bboxes[:, [1]] + head_bboxes[:, [3]]) / 2
                ])    
                sample['head_centers'] = head_centers
                sample["gaze_vecs"] = F.normalize(gaze_pts - head_centers, p=2, dim=-1)
                
                # compute head masks
                _, img_h, img_w = sample['image'].shape
                sample['head_masks'] = generate_mask(head_bboxes, img_w, img_h)

                # generate gaze heatmaps
                sample["gaze_heatmaps"] = generate_gaze_heatmap(gaze_pts, sigma=self.heatmap_sigma, size=self.heatmap_size)

                is_child = torch.zeros(len(heads), dtype=torch.float) - 1
                laeo_ids = lah2laeo(lah_ids)
                coatt_ids = lah2coatt(lah_ids)

                sample['lah_ids'] = lah_ids
                sample['laeo_ids'] = laeo_ids
                sample['coatt_ids'] = coatt_ids
                sample['head_bboxes'] = head_bboxes
                sample['gaze_pts'] = gaze_pts
                sample['heads'] = heads
                sample['inout'] = inout
                sample['speaking'] = speaking_scores
                
                # Append current frame annotations to temporal sample
                t_sample['image'].append(sample['image'])
                t_sample['heads'].append(sample['heads'])
                t_sample['head_centers'].append(sample['head_centers'])
                t_sample['head_masks'].append(sample['head_masks'])
                t_sample['head_bboxes'].append(sample['head_bboxes'])
                t_sample['gaze_pts'].append(sample['gaze_pts'])
                t_sample['gaze_vecs'].append(sample['gaze_vecs'])
                t_sample['gaze_heatmaps'].append(sample['gaze_heatmaps'])
                t_sample['inout'].append(sample['inout'])
                t_sample['lah_ids'].append(sample['lah_ids'])
                t_sample['laeo_ids'].append(sample['laeo_ids'])
                t_sample['coatt_ids'].append(sample['coatt_ids'])
                t_sample['speaking'].append(sample['speaking'])
                t_sample['is_child'].append(is_child)
                t_sample['num_valid_people'].append(sample['num_valid_people'])
                t_sample['img_size'].append(sample['img_size'])
                t_sample['path'].append(path)

        for key, item in t_sample.items():
            if key not in ['dataset', 'path', 'pids']:
                t_sample[key] = torch.stack(t_sample[key], axis=0).squeeze()
                if self.temporal_context==0:
                    t_sample[key] = t_sample[key].unsqueeze(0)
        return t_sample

    def __len__(self):
        return len(self.paths)
        
    
class VideoAttentionTargetDataModule(pl.LightningDataModule):
    def __init__(
        self,
        root: str,
        image_size: Tuple[int, int] = (224, 224),
        batch_size: Union[int, dict] = 32,
        predict_input_file: Union[str, None] = None,
        predict_annotation_file: Union[str, None] = None,
        heatmap_sigma: int = 3,
        heatmap_size = 64,
        num_people: int = 5,
        temporal_context: int=2,
        temporal_stride: int=3
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
        self.temporal_context=temporal_context
        self.temporal_stride=temporal_stride
        self.predict_input_file = predict_input_file
        self.predict_annotation_file = predict_annotation_file
        self.image_size = pair(image_size)


    def setup(self, stage: str):
        if stage == "fit":
            train_transform = Compose(
                [
                    RandomCropSafeGaze(aspect=(self.image_size[0]/self.image_size[1]), p=1.0),
                    ColorJitter(
                        brightness=(0.5, 1.5),
                        contrast=(0.5, 1.5),
                        saturation=(0.0, 1.5),
                        hue=None,
                        p=0.8,
                    ),
                    Resize(img_size=self.image_size, head_size=(224, 224)),
                    ToTensor(),
                    Normalize(
                        img_mean=[0.31072, 0.25703, 0.24182],
                        img_std=[0.26028, 0.24050, 0.23851]
                    ),
                ]
            )
            self.train_dataset = VideoAttentionTargetDataset_temporal(
                root=self.root, 
                split="train", 
                stride=max(3, self.temporal_context*2*3),
                transform=train_transform, 
                tr=(-0.1, 0.1), 
                heatmap_size=self.heatmap_size, 
                heatmap_sigma=self.heatmap_sigma,
                num_people=self.num_people['train'],
                temporal_context=self.temporal_context,
                temporal_stride=self.temporal_stride,
            )

            val_transform = Compose(
                [
                    Resize(img_size=self.image_size, head_size=(224, 224)),
                    ToTensor(),
                    Normalize(
                        img_mean=[0.31072, 0.25703, 0.24182],
                        img_std=[0.26028, 0.24050, 0.23851]
                    ),
                ]
            )
            self.val_dataset = VideoAttentionTargetDataset_temporal(
                root=self.root, 
                split="val", 
                stride=max(3, self.temporal_context*2*3),
                transform=val_transform, 
                tr=(0.0, 0.0), 
                heatmap_size=self.heatmap_size, 
                heatmap_sigma=self.heatmap_sigma,
                num_people=self.num_people['val'],
                temporal_context=self.temporal_context,
                temporal_stride=self.temporal_stride
            )

        elif stage == "validate":
            val_transform = Compose(
                [
                    Resize(img_size=self.image_size, head_size=(224, 224)),
                    ToTensor(),
                    Normalize(
                        img_mean=[0.31072, 0.25703, 0.24182],
                        img_std=[0.26028, 0.24050, 0.23851]
                    ),
                ]
            )
            self.val_dataset = VideoAttentionTargetDataset_temporal(
                root=self.root, 
                split="val",
                stride=max(3, self.temporal_context*2*3),
                transform=val_transform, 
                tr=(0.0, 0.0), 
                heatmap_size=self.heatmap_size, 
                heatmap_sigma=self.heatmap_sigma,
                num_people=self.num_people['val'],
                temporal_context=self.temporal_context,
                temporal_stride=self.temporal_stride
            )

        elif stage == "test":
            test_transform = Compose(
                [
                    Resize(img_size=self.image_size, head_size=(224, 224)),
                    ToTensor(),
                    Normalize(
                        img_mean=[0.31072, 0.25703, 0.24182],
                        img_std=[0.26028, 0.24050, 0.23851]
                    ),
                ]
            )
            self.test_dataset = VideoAttentionTargetDataset_temporal(
                root=self.root, 
                split="test", 
                stride=1,
                transform=test_transform, 
                tr=(0.0, 0.0), 
                heatmap_size=self.heatmap_size, 
                heatmap_sigma=self.heatmap_sigma,
                num_people=self.num_people['test'],
                temporal_context=self.temporal_context,
                temporal_stride=self.temporal_stride
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
            self.predict_dataset = VideoAttentionTargetDataset_temporal(
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
            num_workers=8,
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
