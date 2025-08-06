import os
import random
from glob import glob
from typing import Dict, List, Tuple, Union
import time
from copy import deepcopy
from omegaconf import OmegaConf, ListConfig

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

from src.utils import Stage, pair, expand_bbox, generate_gaze_heatmap, square_bbox, get_ptcloud, CameraToEyeMatrix, generate_mask, load_pkl
from src.transforms import ColorJitter, Compose, Normalize, RandomCropSafeGaze, RandomHeadBboxJitter, RandomHorizontalFlip, Resize, ToTensor

class GazeFollowDataModule(pl.LightningDataModule):
    def __init__(
        self,
        root: str,
        root_depth: Union[str, None] = None,
        root_focal: Union[str, None] = None,
        batch_size: Union[int, dict] = 32,
        image_size = (224, 224),
        heatmap_size = 64,
        num_people: dict = {"train": 1, "val": 1, "test": 1},
        return_depth: bool = False,
        return_head_mask: bool = False,
    ):
        super().__init__()
        self.root = root
        self.root_depth = root_depth
        self.root_focal = root_focal
        if type(image_size)==ListConfig:
            image_size = OmegaConf.to_object(image_size)
        self.image_size = pair(image_size)
        self.heatmap_sigma = int(np.mean(heatmap_size)*3/64)
        self.heatmap_size = heatmap_size
        self.num_people = num_people
        self.batch_size = {stage: batch_size for stage in ["train", "val", "test"]} if isinstance(batch_size, int) else batch_size
        self.return_depth = return_depth
        self.return_head_mask = return_head_mask

    def setup(self, stage: str):
        if stage == "fit":
            train_transform = Compose(
                [
                    RandomCropSafeGaze(aspect=(self.image_size[0]/self.image_size[1]), p=1),
                    RandomHorizontalFlip(p=0.5),
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
                        img_mean=[0.44232, 0.40506, 0.36457],
                        img_std=[0.28674, 0.27776, 0.27995]
                    ),
                ]
            )
            self.train_dataset = GazeFollowDataset(
                self.root,
                self.root_depth,
                self.root_focal,
                "train",
                train_transform,
                tr=(-0.1, 0.1),
                heatmap_size=self.heatmap_size,
                heatmap_sigma=self.heatmap_sigma,
                num_people=self.num_people['train'],
                return_depth=self.return_depth,
                return_head_mask=self.return_head_mask,
            )

            val_transform = Compose(
                [
                    Resize(img_size=self.image_size, head_size=(224, 224)),
                    ToTensor(),
                    Normalize(
                        img_mean=[0.44232, 0.40506, 0.36457],
                        img_std=[0.28674, 0.27776, 0.27995]
                    ),
                ]
            )
            self.val_dataset = GazeFollowDataset(
                self.root,
                self.root_depth,
                self.root_focal,
                "val",
                val_transform,
                tr=(0.0, 0.0),
                heatmap_size=self.heatmap_size,
                heatmap_sigma=self.heatmap_sigma,
                num_people=self.num_people['val'],
                return_depth=self.return_depth,
                return_head_mask=self.return_head_mask,
            )

        elif stage == "validate":
            val_transform = Compose(
                [
                    Resize(img_size=self.image_size, head_size=(224, 224)),
                    ToTensor(),
                    Normalize(
                        img_mean=[0.44232, 0.40506, 0.36457],
                        img_std=[0.28674, 0.27776, 0.27995]
                    ),
                ]
            )
            self.val_dataset = GazeFollowDataset(
                self.root,
                self.root_depth,
                self.root_focal,
                "val",
                val_transform,
                tr=(0.0, 0.0),
                heatmap_size=self.heatmap_size,
                heatmap_sigma=self.heatmap_sigma,
                num_people=self.num_people['val'],
                return_depth=self.return_depth,
                return_head_mask=self.return_head_mask,
            )

        elif stage == "test":
            test_transform = Compose(
                [
                    Resize(img_size=self.image_size, head_size=(224, 224)),   # maintain aspect ratio while testing
                    ToTensor(),
                    Normalize(
                        img_mean=[0.44232, 0.40506, 0.36457],
                        img_std=[0.28674, 0.27776, 0.27995]
                    ),
                ]
            )
            self.test_dataset = GazeFollowDataset(
                self.root,
                self.root_depth,
                self.root_focal,
                "test",
                test_transform,
                tr=(0.0, 0.0),
                heatmap_size=self.heatmap_size,
                heatmap_sigma=self.heatmap_sigma,
                num_people=self.num_people['test'],
                return_depth=self.return_depth,
                return_head_mask=self.return_head_mask,
            )

        elif stage == "predict":
            predict_transform = Compose(
                [
                    Resize(img_size=self.image_size, head_size=(224, 224)),
                    ToTensor(),
                    Normalize(
                        img_mean=[0.44232, 0.40506, 0.36457],
                        img_std=[0.28674, 0.27776, 0.27995]
                    ),
                ]
            )
            self.predict_dataset = GazeFollowDataset(
                self.root,
                self.root_depth,
                self.root_focal,
                "test",
                predict_transform,
                tr=(0.0, 0.0),
                heatmap_size=self.heatmap_size,
                heatmap_sigma=self.heatmap_sigma,
                num_people='all',
                return_depth=self.return_depth,
                return_head_mask=self.return_head_mask,
            )

    def train_dataloader(self):
        dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size["train"],
            shuffle=True,
            num_workers=14,
            pin_memory=True,
        )
        return dataloader

    def val_dataloader(self):
        dataloader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size["val"],
            shuffle=False,
            num_workers=6,
            pin_memory=True,
        )
        return dataloader

    def test_dataloader(self):
        dataloader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size["test"],
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )
        return dataloader

    def predict_dataloader(self):
        dataloader = DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size["predict"],
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )
        return dataloader


# ============================================================================= #
#                               GAZEFOLLOW DATASET                              #
# ============================================================================= #
class GazeFollowDataset(Dataset):
    def __init__(
        self,
        root,
        root_depth = None,
        root_focal = None,
        split: str = "train",
        transform: Union[Compose, None] = None,
        tr: tuple = (-0.1, 0.1),
        heatmap_sigma: int = 3,
        heatmap_size: int = 64,
        num_people: int = 5,
        head_thr: float = 0.5,
        return_depth: bool = False,
        return_head_mask: bool = False,
    ):
        super().__init__()

        assert split in ("train", "val", "test"), f"Expected `split` to be one of [`train`, `val`, `test`] but received `{split}` instead."
        assert (num_people == 'all') or (num_people > 0), f"Expected `num_people` to be strictly positive or \"all\", but received {num_people} instead."
        assert 0 <= head_thr <= 1, f"Expected `head_thr` to be in [0, 1]. Received {head_thr} instead."

        self.root = root
        self.root_depth = root_depth
        self.root_focal = root_focal
        self.split = split
        self.jitter_bbox = RandomHeadBboxJitter(p=1.0, tr=tr)
        self.transform = transform
        self.heatmap_sigma = heatmap_sigma
        self.heatmap_size = heatmap_size
        self.num_people = num_people
        self.head_thr = head_thr
        self.return_depth = return_depth
        self.return_head_mask = return_head_mask
        self.annotations, self.paths = self.load_annotations()
        self.length = len(self.paths)
        
#         # ======== for VLM context =========
# #         path_to_vlm = '/idiap/temp/pvuillecard/projects/nlp_vlm/results/gazefollow/gazefollow_ensemble.pkl' # AVA + CP
#         path_to_vlm = '/idiap/temp/pvuillecard/projects/nlp_vlm/experiments/2024-03-01/15-41-58/outputs/gazefollow_score_blip2_ensemble_text_prompt_hico.pkl'
# #         path_to_vlm = '/idiap/temp/pvuillecard/projects/nlp_vlm/experiments/2024-03-01/15-42-28/outputs/gazefollow_score_blip2_ensemble_text_prompt_swig.pkl'
#         loaded_vlm_context = load_pkl(path_to_vlm)
#         self.loaded_vlm_context = loaded_vlm_context
# #         self.loaded_vlm_context = {f"{item['path']}_{item['split']}": item for item in loaded_vlm_context}
#         del loaded_vlm_context

    def load_vlm_context(self, path, split):
        key = f"{path}_{split}"
        if key in self.loaded_vlm_context:
            return self.loaded_vlm_context[key]
        raise FileNotFoundError(f'Unable to find vlm pkl file of path {path} and split {split}')

    def load_annotations(self) -> pd.DataFrame:
        annotations = pd.DataFrame()
        if self.split == "test":
            column_names = ["path", "id", "body_bbox_x", "body_bbox_y", "body_bbox_w", "body_bbox_h", "eye_x", "eye_y", 
                            "gaze_x", "gaze_y", "head_xmin", "head_ymin", "head_xmax", "head_ymax", "origin", "meta"]
            annotations = pd.read_csv(
                os.path.join(self.root, "test_annotations_release.txt"),
                sep=",",
                names=column_names,
                index_col=False,
                encoding="utf-8-sig",
            )
            # Add inout col for consistency (ie. missing from test set)
            annotations["inout"] = 1
            # Each test image is annotated by multiple people (around 10 on avg.)
            # group by path
            annotations = annotations.groupby('path')
            paths = list(annotations.groups.keys())

        elif self.split in ["train", "val"]:
            column_names = ["path", "id", "body_bbox_x", "body_bbox_y", "body_bbox_w", "body_bbox_h", "eye_x", "eye_y", 
                            "gaze_x", "gaze_y", "head_xmin", "head_ymin", "head_xmax", "head_ymax", "inout", "origin", "meta"]
            annotations = pd.read_csv(
                os.path.join(f"/idiap/temp/agupta/data/attention/gazefollow/{self.split}_annotations_new.txt"),
                sep=",",
                names=column_names,
                index_col=False,
                encoding="utf-8-sig",
            )
            
            # Clean annotations (e.g. remove invalid ones)
            annotations = self._clean_annotations(annotations)
            
            # group by path
            annotations = annotations.groupby('path')
            paths = list(annotations.groups.keys())

        return annotations, paths

    def _clean_annotations(self, annotations):
        # Only keep "in" and "out". (-1 is invalid)
        annotations = annotations[annotations.inout != -1]
        # Discard instances where max in bbox coordinates is smaller than min
        annotations = annotations[annotations.head_xmin < annotations.head_xmax]
        annotations = annotations[annotations.head_ymin < annotations.head_ymax]
        return annotations.reset_index(drop=True)

    def __getitem__(self, index: int) -> Dict:
        
        # load annotations
        path = self.paths[index]
        img_annotations = self.annotations.get_group(path)
        inout = torch.from_numpy(img_annotations["inout"].values.astype(np.float32))
        gaze_pts = torch.from_numpy(img_annotations[["gaze_x", "gaze_y"]].values.astype(np.float32))
        inout = torch.from_numpy(img_annotations["inout"].values.astype(np.float32))
        
        if self.split in ["train", "val"]:
            idx = img_annotations["id"].values.astype(np.float32)
        elif self.split == "test":
            p = 20 - len(gaze_pts)
            # Pad to have same length across samples for dataloader
            gaze_pts = F.pad(gaze_pts, (0, 0, 0, p), value=-1.0)
            idx = img_annotations["id"].values.tolist() + [-1] * p  # pad to 20 for consistency
            img_annotations = img_annotations[:1]
            inout = inout[0]

        # Load image
        image = Image.open(os.path.join(self.root, path)).convert("RGB")
        img_w, img_h = image.size
        gaze_pts = gaze_pts * torch.tensor([img_w, img_h])
        
        pcd = torch.zeros((3, img_h, img_w), dtype=torch.float32)
        if self.return_depth:
            # Load focal length, depth map
            focal_path = os.path.join(self.root_focal, path[:-4]+'-focal_length.txt')
            focal_length = float(open(focal_path).read().split('\n')[0])
            depth = torch.load(os.path.join(self.root_depth, path[:-4]+'.pth')).float().unsqueeze(0)
            depth = TF.resize(depth, (img_h, img_w), antialias=True)
            # Compute point cloud
            pcd = get_ptcloud(depth.squeeze(0), focal_length).permute(2,0,1)

        # Load head bboxes
        ## For annotated people
        head_bboxes = img_annotations[["head_xmin", "head_ymin", "head_xmax", "head_ymax"]]
        head_bboxes = torch.from_numpy(head_bboxes.values.astype(np.float32))
        head_bboxes = expand_bbox(head_bboxes, img_w, img_h, k=0.1)    # expand bbox slightly

        # Shuffle annotated people
        if self.split == "train":
            rand_indices = torch.randperm(head_bboxes.size(0))
            head_bboxes = head_bboxes[rand_indices]
            gaze_pts = gaze_pts[rand_indices]
            inout = inout[rand_indices]
        
        # Get detected head bboxes
        split, folder, img_name = path.split("/")
        basename, ext = os.path.splitext(img_name)
        det_file = f"{split}/{folder}/{basename}-head-detections.npy"
        det_head_bboxes = np.load(os.path.join("/idiap/temp/stafasca/data/GazeFollow-head", det_file))
        
        # Process detected head bboxes
        if len(det_head_bboxes) > 0:
            scores = torch.tensor(det_head_bboxes[:, -1])
            det_head_bboxes = torch.tensor(det_head_bboxes[(scores >= self.head_thr).tolist(), :-1]).float()
            # merge annotated head bboxes
            ious = box_iou(det_head_bboxes, head_bboxes)
            ious, index_ious = torch.max(ious, axis=1)
            index_keep = ious < 0.3
            det_head_bboxes = det_head_bboxes[index_keep]
            # Shuffle detected people
            if self.split == "train":
                rand_indices = torch.randperm(det_head_bboxes.size(0))
                det_head_bboxes = det_head_bboxes[rand_indices]
            head_bboxes = torch.concat([det_head_bboxes, head_bboxes], axis=0)
            # concat -1 for gaze and inout annotations
            if self.split!='test':
                gaze_pts = torch.cat([torch.zeros((len(det_head_bboxes), 2))-1, gaze_pts])
                inout = torch.cat([torch.zeros(len(det_head_bboxes), dtype=torch.float32)-1, inout])
                
#         #--------------------------------------------------------------------------------
#         # Match VLM scores with people
#         vlm_context = self.loaded_vlm_context[path]
#         p_vlm_context = torch.from_numpy(vlm_context['prompt_score']) # (n, C_person)
#         head_bbox_vlm = torch.from_numpy(vlm_context['bbox_head']) # (n, 4)
# #         vlm_context = self.load_vlm_context(path, self.split)
# #         assert path==vlm_context['path'], 'found a mismatch between path and VLM path'
# #         p_vlm_context = torch.from_numpy(vlm_context['person_scores']) # (n, C_person)
# #         head_bbox_vlm = torch.from_numpy(vlm_context['head_bboxes']) # (n, 4)
#         dim_vlm = p_vlm_context.shape[1]
#         # Perform matching with vlm head bbox
#         ious_vlm = box_iou(head_bboxes, head_bbox_vlm)
#         max_iou_values, max_iou_indices = torch.max(ious_vlm, dim=1)
#         person_vlm_context = torch.zeros((len(head_bboxes), dim_vlm), dtype=torch.float) - 1
#         for mi, vlm_i in enumerate(max_iou_indices):
#             if max_iou_values[mi]>0.5:
#                 person_vlm_context[mi] = p_vlm_context[vlm_i]
#         #--------------------------------------------------------------------------------
        
        if self.split == "train":
            head_bboxes = self.jitter_bbox(head_bboxes, img_w, img_h)

        # Square head bboxes (can have negative values)
        head_bboxes = square_bbox(head_bboxes, img_w, img_h)

        # Extract Heads (negative values add padding)
        heads = []
        for head_bbox in head_bboxes:
            heads.append(image.crop(head_bbox.int().tolist()))  # type:ignore
   
        # Select {1, num_people} people
        num_heads = len(heads)
        num_keep = num_heads
        if self.num_people!='all':
            if num_heads>1:
                num_keep = np.random.randint(2, min(num_heads, self.num_people)+1)   # min 2 people
#                 num_keep = np.random.randint(1, min(num_heads, self.num_people)+1)   # min one person
        head_bboxes = head_bboxes[-num_keep:]
        heads = heads[-num_keep:]
        # person_vlm_context = person_vlm_context[-num_keep:]
        if self.split!='test':
            gaze_pts = gaze_pts[-num_keep:]
            inout = inout[-num_keep:]
        num_heads = len(heads)
#         num_missing_heads = max(self.num_people - num_heads, 0) if self.num_people != "all" else 0    
        num_missing_heads = max(self.num_people + 1 - num_heads, 1) if self.num_people != "all" else 1    # pad at least one person
        
        # Get lah id
        lah_ids = torch.zeros(len(heads), dtype=torch.long) - 5
        if self.split!='test':
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
        else:
            lah_id_prev = -5
            for i in range(len(gaze_pts)):
                gaze_pt = gaze_pts[i]
                io = inout
                if io==0:
                    lah_ids[len(heads)-1] = -1
                elif io==1:
                    # check if gaze point is inside any head bbox
                    inside1 = (head_bboxes[:, :2] < gaze_pt).int().prod(1)
                    inside2 = (head_bboxes[:, 2:] > gaze_pt).int().prod(1)
                    lah_id = np.where(inside1 * inside2)[0]
                    if len(lah_id)>=1:
                        lah_id = lah_id[:1].item()
                        if lah_id_prev == lah_id:
                            if lah_id==(len(heads)-1):
                                lah_id = -1
                            lah_ids[len(heads)-1] = lah_id
                            break
                        else:
                            lah_id_prev = lah_id
                    else:
                        lah_ids[len(heads)-1] = -1
            
        # Create (Normalized) Gaze Points
        gaze_pts[gaze_pts[:, 0] != -1.] /= torch.tensor([img_w, img_h])
            
        # Pad missing people (ie. heads, head_bboxes, gaze_pt and coatt); always have one extra person
        if num_missing_heads > 0:
            head_bboxes = torch.cat([torch.zeros((num_missing_heads, 4)), head_bboxes])
            heads = num_missing_heads * [Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))] + heads
            gaze_pts = torch.cat([torch.zeros((num_missing_heads, 2)) - 1, gaze_pts])
            lah_ids[lah_ids>=0] += num_missing_heads
            lah_ids = torch.cat([torch.zeros((num_missing_heads, ), dtype=torch.long)-3, lah_ids])
            # person_vlm_context = torch.cat([torch.zeros((num_missing_heads, person_vlm_context.shape[1]), dtype=torch.float32)-1, person_vlm_context])
        if self.split!='test':
            inout = torch.cat([torch.zeros((num_missing_heads, ), dtype=torch.float32)-1, inout])
        
        # Normalize Head Bboxes
        head_bboxes /= torch.tensor([img_w, img_h, img_w, img_h], dtype=float)
            
        # Build Sample
        sample = {
                "image": image,
                "pcd": pcd,
                "heads": heads,
                "head_bboxes": head_bboxes,
                "gaze_pts": gaze_pts,
                "inout": inout,
                "coatt_ids": torch.zeros(len(heads), dtype=torch.long),
                "lah_ids": lah_ids,
                "laeo_ids": torch.zeros(len(heads), dtype=torch.long),
                "num_valid_people": torch.tensor(num_heads),
                "is_child": torch.zeros(len(heads), dtype=torch.float) - 1,
                "speaking": torch.zeros(len(heads), dtype=torch.float) - 1,
                "img_size": torch.tensor((img_w, img_h), dtype=torch.long),
                "path": [path],
                "dataset": 'gazefollow',
                # "person_vlm_context": person_vlm_context
        }
        
        # Transform
        if self.transform:
            sample = self.transform(sample)
            
        if not self.return_depth:
            sample.pop('pcd')
            
        # compute head masks
        _, img_h, img_w = sample['image'].shape
        sample['head_masks'] = generate_mask(sample['head_bboxes'], img_w, img_h)
            
         # Compute Head Bbox Centers
        head_bboxes = sample['head_bboxes']
        gaze_pts = sample['gaze_pts']
        head_centers = torch.hstack([
            (head_bboxes[:, [0]] + head_bboxes[:, [2]]) / 2, 
            (head_bboxes[:, [1]] + head_bboxes[:, [3]]) / 2
        ])    
        sample['head_centers'] = head_centers

        # compute 2d gaze vectors
        if self.split=='test':
            head_centers = head_centers[-1].unsqueeze(0)   # only consider annotated person
        sample["gaze_vecs"] = F.normalize(gaze_pts - head_centers, p=2, dim=-1)
        
        if self.return_depth:
            pcd = sample['pcd'].permute(1,2,0)
            # compute 3d gaze vectors
            gaze_pts = (gaze_pts* torch.tensor([img_w, img_h])).int()
            head_centers = (head_centers* torch.tensor([img_w, img_h])).int()
            head_centers_3d = pcd[head_centers[:,1],head_centers[:,0]]
            gaze_vecs_3d = F.normalize(pcd[gaze_pts[:,1], gaze_pts[:,0]] - head_centers_3d, p=2, dim=-1)  # gaze vecs in camera coordinate system
            dirEyes = F.normalize(head_centers_3d, p=2, dim=-1)
            cam2eye = CameraToEyeMatrix(dirEyes, sample['inout'])  # get camera to eye coordinate matrix
            gaze_vecs_3d = torch.matmul(cam2eye, gaze_vecs_3d.unsqueeze(-1))  # gaze vecs in eye coordinate system
            sample['gaze_vecs_3d'] = gaze_vecs_3d.squeeze(-1)
            sample['cam2eye'] = cam2eye
            
        if self.split!='test':
            # generate gaze heatmaps
            sample["gaze_heatmaps"] = generate_gaze_heatmap(sample['gaze_pts'], sigma=self.heatmap_sigma, size=self.heatmap_size)
            
        # add extra dimension to be compatible with temporal model
        for key, item in sample.items():
            if key not in ['path', 'dataset']:
                sample[key] = item.unsqueeze(0)

        return sample

    def __len__(self):
        return self.length
