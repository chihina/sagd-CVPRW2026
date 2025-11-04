import os
import random
import sys
from glob import glob
from typing import Dict, List, Tuple, Union
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
from src.utils import square_bbox, generate_gaze_heatmap, Stage, lah2laeo, lah2coatt, get_ptcloud, CameraToEyeMatrix, generate_mask, pair, load_pkl, generate_coatt_heatmap

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


# =============================================================================
#                              ChildPlay Dataset
# =============================================================================
class ChildPlayDataset_temporal(Dataset):
    def __init__(
        self,
        cfg,
        root: str,
        root_depth = None,
        root_focal = None,
        split: str = "train",
        stride: int = 3,
        transform: Union[Compose, None] = None,
        tr=(-0.1, 0.1),
        image_size=(224,224),
        heatmap_sigma=3,
        heatmap_size=64,
        num_people=5,
        subset='full',    # full, child, adult
        temporal_stride = 3,
        temporal_context = 2,
        return_depth: bool = False,
        aspect=False,
        dim_vlm = 117
    ):
        super().__init__()
        self.cfg = cfg
        self.root = root
        self.root_depth = root_depth
        self.root_focal = root_focal
        self.split = split
        self.stride = stride
        self.jitter_bbox = RandomHeadBboxJitter(p=1.0, tr=tr)
        self.transform = transform
        self.image_size = image_size
        self.heatmap_sigma = heatmap_sigma
        self.heatmap_size = heatmap_size
        self.num_people = num_people
        self.subset = subset
        self.head_thr = 0.5
        self.temporal_stride = temporal_stride
        self.temporal_context = temporal_context
        self.pid_offset = 1000
        self.return_depth = return_depth
        self.aspect=aspect
        self.dim_vlm = dim_vlm
        self.return_vlm_context = False
        self.return_speaking_features = False
        self.num_coatt = cfg.data.num_coatt

        # load annotations
        # self.annotations, self.keys = self._load_annotations()
        # self.annotations = self._load_annotations()
        self.annotations, self.paths = self._load_annotations()
        
        paths_orignal = self.paths
        self.paths = [p for p in self.paths if self.valid_data(p)]
        print(f'[ChildPlay]: {len(self.paths)}/{len(paths_orignal)} valid data in {self.split} set')

         # load speaking status file
        # self.df_speaking = self.load_annotations_speaker()
        # self.df_speaking = self.df_speaking.groupby('path')
        
        # load speaking features
        # self.df_speaking_features = self.load_features_speaker()
        # self.df_speaking_features = self.df_speaking_features.groupby('path')
        
        # ======== for VLM context =========
        path_to_vlm = '/idiap/temp/pvuillecard/projects/nlp_vlm/experiments/2024-03-14/17-59-47/outputs/childplay_score_blip2_ensemble_text_prompt_swig.pkl'  # SWIG
#         path_to_vlm = '/idiap/temp/pvuillecard/projects/nlp_vlm/experiments/2024-03-14/18-00-15/outputs/childplay_score_blip2_ensemble_text_prompt_ava.pkl'  # AVA + CP
#         path_to_vlm = '/idiap/temp/pvuillecard/projects/nlp_vlm/experiments/2024-02-26/16-18-28/outputs/childplay_score_blip2_ensemble_text_prompt_hico.pkl'  # hico
        # loaded_vlm_context = load_pkl(path_to_vlm)
        # self.loaded_vlm_context = loaded_vlm_context
        # del loaded_vlm_context

    def valid_data(self, path):
        # clip = path.split('/')[1]
        # video_id, interval = clip.rsplit('_', 1)
        # frame = int(path.split('/')[-1].split('.')[0].split('_')[-1])
        # curr_frame_nb = frame
        # frame_nbs = np.arange(curr_frame_nb-(self.temporal_stride*self.temporal_context), curr_frame_nb+(self.temporal_stride*self.temporal_context)+1, self.temporal_stride)
        # for f in frame_nbs:
        #     img_name = f'{video_id}_{f}.jpg'
        #     path = os.path.join('images', clip, img_name)
        #     if not os.path.exists(os.path.join(self.root, path)):
        #         return False

        # return True

        return os.path.exists(os.path.join(self.root, path))

    def load_annotations_speaker(self):
        annotation_files = glob(os.path.join('/idiap/temp/agupta/data/child-play/speaker_corr/', f"*.csv"))

        li = []
        for file in annotation_files:
            clip = file.split('/')[-1]
            clip = clip.split('.')[0]
            df = pd.read_csv(file)

            # add column for path
            frame_names = glob(os.path.join(self.root, 'images', clip, '*.jpg'))
            frame_names.sort(key=lambda f: int(f.split('_')[-1][:-4]))
            paths = [os.path.join('images', clip, frame_names[int(f) - 1].split('/')[-1]) for f in df['frame'].values]
            df['path'] = paths

            li.append(df)
        annotations = pd.concat(li, axis=0, ignore_index=True)

        return annotations
    
    def load_features_speaker(self):
        annotation_files = glob(os.path.join('/idiap/temp/agupta/data/child-play/speaking_features/', f"*.csv"))

        li = []
        for file in annotation_files:
            clip = file.split('/')[-1]
            clip = clip.split('.')[0]
            df = pd.read_csv(file, names=['clip', 'frame', 'id', 'na_1', 'na_2', 'na_3', 'xmin', 'ymin', 'xmax', 'ymax', 'feat'])

            # add column for path
            frame_names = glob(os.path.join(self.root, 'images', clip, '*.jpg'))
            frame_names.sort(key=lambda f: int(f.split('_')[-1][:-4]))
            paths = [os.path.join('images', clip, frame_names[int(f) - 1].split('/')[-1]) for f in df['frame'].values]
            df['path'] = paths

            li.append(df)
        annotations = pd.concat(li, axis=0, ignore_index=True)

        return annotations
        
    # exclude_cls=['gaze_shift', 'inside_occluded', 'inside_uncertain', 'eyes_closed']
    def _load_annotations(self, exclude_cls=[]):
        # files = glob(os.path.join(self.root, 'annotations', self.split, '*.csv'))
        # li = []
        # for file in files:
        #     df = pd.read_csv(file)
        #     li.append(df)
        # annotations = pd.concat(li, axis=0, ignore_index=True)

        annotation_path = f"data/VSGaze/childplay_{self.split}.h5"
        annotations = pd.read_hdf(annotation_path)
        
        # if len(exclude_cls) > 0:
            # cond = annotations.gaze_class.isin(exclude_cls)
            # annotations = annotations[~cond].reset_index(drop=True)
        
        # Temporarily remove extra annotation for which there is no frame
        # annotations = annotations.drop(annotations[(annotations['clip'] == '4yWavYq9_Ks_405-451') & (annotations.frame == 48)].index)
        
        # Change cases where gaze_class=='inside_visible' but gaze_x==-1 to gaze_class=='inside-uncertain'
        # incorr_indices = annotations[(annotations['gaze_class']=='inside_visible') & (annotations['gaze_x'] == -1)].index
        # annotations['gaze_class'][incorr_indices] = 'inside_uncertain'
        
        # Drop children or adults from the dataset
        # print('ChildPlay subset: ', self.subset)
        # if self.subset=='child':
            # annotations = annotations.drop(annotations[(annotations['is_child']==0)].index)
        # elif self.subset=='adult':
            # annotations = annotations.drop(annotations[(annotations['is_child']==1)].index)
        
        # re-name head bbox annotations
        # annotations = annotations.rename(columns={'bbox_x': 'head_xmin', 'bbox_y': 'head_ymin'})
        # annotations['head_xmax'] = annotations['head_xmin'] + annotations['bbox_width']
        # annotations['head_ymax'] = annotations['head_ymin'] + annotations['bbox_height']
        
        # merge with speaking status annotations
        # df_gt_speaking = pd.read_csv('/idiap/temp/agupta/data/child-play/childplay_speaking.csv', index_col=0)
        # annotations = annotations.merge(df_gt_speaking, on=['clip', 'frame', 'person_id'], how='left')
        
        # group by clip and frame
        # annotations = annotations.groupby(['clip', 'frame'])
        # keys = np.array(list(annotations.groups.keys()))
        
        # if self.stride > 1:
            # index_keep = np.arange(len(keys), step=self.stride)
            # keys = keys[index_keep]
        
        annotations = annotations.groupby('path')
        paths = list(annotations.groups.keys())
        paths = np.array(paths)

        if self.stride > 1:
            index_keep = np.arange(len(paths), step=self.stride)
            paths = paths[index_keep]

        # return annotations, keys
        return annotations, paths
        
    def __getitem__(self, index):
        # clip, frame = self.keys[index]
        
        # frame = int(frame)
        # jitter frame number during training
        # if self.split=='train':
        #     if self.temporal_context==0:
        #         frame_shift = torch.randint(-(self.temporal_stride//2), self.temporal_stride//2, (1,)).item()
        #     else:
        #         frame_shift = torch.randint(-(self.temporal_context*self.temporal_stride-1), self.temporal_context*self.temporal_stride, (1,)).item()
        #     frame_tmp = frame + frame_shift
        #     while (clip, frame_tmp) not in self.annotations.groups.keys():
        #         if self.temporal_context==0:
        #             frame_shift = torch.randint(-(self.temporal_stride//2), self.temporal_stride//2, (1,)).item()
        #         else:
        #             frame_shift = torch.randint(-(self.temporal_context*self.temporal_stride-1), self.temporal_context*self.temporal_stride, (1,)).item()
        #         frame_tmp = frame + frame_shift
        #     frame = frame_tmp
        
        # img_annotations = self.annotations.get_group((clip, frame))

        path = self.paths[index]
        img_annotations = self.annotations.get_group(path)
        frame = int(path.split('/')[-1].split('.')[0].split('_')[-1])
        curr_frame_nb = frame

        # get current frame num
        # clip = img_annotations['clip'].values[0]
        # video_id, interval = clip.replace('-downsample', '').rsplit('_', 1)

        clip = path.split('/')[1]
        video_id, interval = clip.rsplit('_', 1)
        # offset = int(interval.split('-')[0])
        # print(video_id, interval, clip, offset, curr_frame_nb, path)

        # curr_frame_nb = img_annotations['frame'].values[0]
        # img_name = f'{video_id}_{offset + curr_frame_nb - 1}.jpg'
        # path = os.path.join('images', clip, img_name)
        
        # read current frame
        img_path = os.path.join(self.root, path)
        image = Image.open(img_path)  
        img_w, img_h = image.size
        if self.split=='test' and self.aspect:    # for maintaining aspect ratio
            dummy_sample = {}
            dummy_sample['image'] = image
            dummy_sample['heads'] = []
            dummy_sample['pcd'] = torch.zeros((3,img_h,img_w), dtype=torch.float32)
            dummy_sample = Resize(img_size=self.image_size[1], head_size=(224,224))(dummy_sample)
            self.image_size = dummy_sample['image'].size
        
        # get frame nums around current frame
        frame_nbs = np.arange(curr_frame_nb-(self.temporal_stride*self.temporal_context), curr_frame_nb+(self.temporal_stride*self.temporal_context)+1, self.temporal_stride)
        head_bboxes = img_annotations['head_bboxes']
        head_bboxes = torch.from_numpy(head_bboxes.values[0].astype(np.float32))

        pids_ann = np.arange(len(head_bboxes))

        # if self.split=='train':    # shuffle pids
        if self.split in ['train', 'val']:    # shuffle pids
            rand_indices = torch.randperm(head_bboxes.size(0))
            pids_ann = pids_ann[rand_indices]
        if len(pids_ann.shape)==0:
            pids_ann = np.expand_dims(pids_ann, 0)

        person_ids = pids_ann
        num_heads = len(person_ids)
        num_keep = num_heads

        if self.num_people!='all':
            batch_num_heads = self.num_people
            if num_heads>1:
                # num_keep = np.random.randint(2, min(num_heads, self.num_people)+1)
                num_keep = min(num_heads, self.num_people)
        else:
            batch_num_heads = num_heads
        
        '''
        if self.num_people!='all':
            batch_num_heads = self.num_people
        else:
            batch_num_heads = num_heads
        '''

        person_ids = person_ids[:num_keep]
        
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
                "coatt_heatmaps": [],
                "coatt_levels": [],
                "lah_ids": [],
                "laeo_ids": [],
                "coatt_ids": [],
                "speaking": [],
                "is_child": [],
                "num_valid_people": [],
                "img_size": [],
                "path": [],
                "dataset": 'childplay'
                }
        t_sample['pids'] = torch.cat([torch.zeros((batch_num_heads + 1 -len(person_ids), ))-1, torch.from_numpy(person_ids)])
        if self.return_vlm_context:
            t_sample["person_vlm_context"] = []
        if self.return_depth:
            t_sample['pcd'] = []
            t_sample['gaze_vecs_3d'] = []
            t_sample['cam2eye'] = []
        if self.return_speaking_features:
            t_sample['speaking_features'] = []
        for frame_nb in frame_nbs:
            # check if frame exists
            # if not (clip, frame_nb) in self.annotations.groups.keys():
            img_name = f'{video_id}_{frame_nb}.jpg'
            path = os.path.join('images', clip, img_name)
            if not path in self.annotations.groups.keys():
                t_sample['image'].append(torch.zeros((3, self.image_size[1], self.image_size[0]), dtype=torch.float32))
                t_sample['heads'].append(torch.zeros((batch_num_heads+1, 3, 224, 224), dtype=torch.float32))
                t_sample['head_centers'].append(torch.zeros((batch_num_heads+1, 2), dtype=torch.float32))
                t_sample['head_masks'].append(torch.zeros((batch_num_heads+1, 1, self.image_size[1], self.image_size[0]), dtype=torch.float32))
                t_sample['head_bboxes'].append(torch.zeros((batch_num_heads+1, 4), dtype=torch.float32))
                t_sample['gaze_pts'].append(torch.zeros((batch_num_heads+1, 2), dtype=torch.float32)-1)
                t_sample['gaze_vecs'].append(torch.zeros((batch_num_heads+1, 2), dtype=torch.float32))
                t_sample['gaze_heatmaps'].append(torch.zeros((batch_num_heads+1, self.heatmap_size, self.heatmap_size), dtype=torch.float32))
                t_sample['coatt_heatmaps'].append(torch.zeros((self.num_coatt, self.heatmap_size, self.heatmap_size), dtype=torch.float32))
                t_sample['coatt_levels'].append(torch.zeros((self.num_coatt, batch_num_heads+1), dtype=torch.int))
                t_sample['inout'].append(torch.zeros((batch_num_heads+1), dtype=torch.float32)-1)
                t_sample['lah_ids'].append(torch.zeros((batch_num_heads+1), dtype=torch.long)-3)
                t_sample['laeo_ids'].append(torch.zeros((batch_num_heads+1), dtype=torch.long))
                t_sample['coatt_ids'].append(torch.zeros((batch_num_heads+1), dtype=torch.long))
                t_sample['speaking'].append(torch.zeros((batch_num_heads+1), dtype=torch.float32)-1)
                t_sample['is_child'].append(torch.zeros((batch_num_heads+1), dtype=torch.float32)-1)
                t_sample['num_valid_people'].append(torch.zeros(1, dtype=torch.long))
                t_sample['img_size'].append(torch.zeros(2, dtype=torch.long))
                t_sample['path'].append('')
                if self.return_vlm_context:
                    t_sample['person_vlm_context'].append(torch.zeros((batch_num_heads + 1), self.dim_vlm))
                if self.return_depth:
                    t_sample['pcd'].append(torch.zeros((3, self.image_size[1], self.image_size[0]), dtype=torch.float32))
                    t_sample['gaze_vecs_3d'].append(torch.zeros((batch_num_heads+1, 3), dtype=torch.float32))
                    t_sample['cam2eye'].append(torch.zeros((batch_num_heads+1, 3, 3), dtype=torch.float32))
                if self.return_speaking_features:
                    t_sample['speaking_features'].append(torch.zeros((batch_num_heads + 1), 512))
            else:
                ###########################################
                # Get annotations
                ###########################################
                # Load image
                # img_name = f'{video_id}_{offset + frame_nb - 1}.jpg'
                img_name = f'{video_id}_{frame_nb}.jpg'
                path = os.path.join('images', clip, img_name)
                img_path = os.path.join(self.root, path)
                image = Image.open(img_path)  
                img_w, img_h = image.size

                pcd = torch.zeros((3, img_h, img_w), dtype=torch.float32)
                if self.return_depth:
                    # Load focal length, depth map
                    focal_path = os.path.join(self.root_focal, clip, img_name[:-4]+'-focal_length.txt')
                    focal_length = float(open(focal_path).read().split('\n')[0])
                    depth = torch.load(os.path.join(self.root_depth, clip, img_name[:-4]+'.pth')).float().unsqueeze(0)
                    depth = TF.resize(depth, (img_h, img_w), antialias=True)
                    # Compute point cloud
                    pcd = get_ptcloud(depth.squeeze(0), focal_length).permute(2,0,1)

                det_head_bboxes = []

                # Load annotations for selected person ids
                img_annotations = self.annotations.get_group(path)

                # pids_ann = img_annotations["person_id"].values

                head_bbox_img = img_annotations['head_bboxes']
                head_bbox_img = torch.from_numpy(head_bbox_img.values[0].astype(np.float32))
                gaze_pt_img = img_annotations['gaze_points']
                gaze_pt_img = torch.from_numpy(gaze_pt_img.values[0].astype(np.float32))
                inout_img = img_annotations['inout']
                inout_img = torch.from_numpy(inout_img.values[0].astype(np.float32))

                pids_ann = np.arange(len(head_bboxes))
                person_ids_ann = np.unique(pids_ann)

                head_bboxes = []; gaze_pts = []; inout = []; gt_speaking = []; speaking_scores = []; is_child = []
                coatt_ids = []

                # Load annotations of coatt pairs and generate coatt ids
                coatt_pairs = img_annotations['coatt_pairs'].values[0]
                pairs = img_annotations['pairs'].values[0]
                pids = img_annotations["person_ids"].values[0]
                pid2idx = {pid: i for i, pid in enumerate(pids)}
                idx2pid = {i: pid for i, pid in enumerate(pids)}
                pairs = [(pid2idx[i], pid2idx[j]) for i,j in pairs]

                # during training, select fixed number of people for faster training
                # if self.split=='train' and self.num_people != 'all':
                if self.split in ['train', 'val'] and self.num_people != 'all':
                    pids_pad = pids[0]
                    pids_wo_pad = pids[1:]
                    pids_wo_pad = np.random.permutation(pids_wo_pad)[:num_keep-1]
                    pids = np.concatenate(([pids_pad], pids_wo_pad))

                coatt_p_ids = {}
                for pair_idx, coatt_pair in enumerate(coatt_pairs):
                    if coatt_pair == 1:
                        pair = pairs[pair_idx]
                        pid_1, pid_2 = map(int, pair)

                        if not (idx2pid[pid_1] in pids and idx2pid[pid_2] in pids):
                            continue
                        
                        find_coatt_id = False
                        for coatt_id, p_ids in coatt_p_ids.items():
                            if (pid_1 in p_ids) or (pid_2 in p_ids):
                                coatt_p_ids[coatt_id].add(pid_1)
                                coatt_p_ids[coatt_id].add(pid_2)
                                find_coatt_id = True
                        if not find_coatt_id:
                            coatt_p_ids[len(coatt_p_ids)+1] = set([pid_1, pid_2])

                for pid in sorted(pids):
                    pid_idx = pid2idx[pid]
                    if pid_idx == 0:
                        head_bboxes.append(torch.zeros(4, dtype=torch.float32))
                        gaze_pts.append(torch.zeros(2, dtype=torch.float32)-1)
                        inout.append(-1)
                        coatt_ids.append(0)
                        gt_speaking.append(-1)
                        speaking_scores.append(-1)
                        is_child.append(-1)
                    else:
                        head_bbox = head_bbox_img[pid_idx].squeeze()
                        head_bboxes.append(head_bbox)

                        gaze_pt = gaze_pt_img[pid_idx].squeeze()
                        gaze_pts.append(gaze_pt)
                        
                        is_child.append(-1)
                        io = inout_img[pid_idx].squeeze().item()
                        inout.append(io)
                        
                        # get coatt id
                        find_coatt_id = False
                        for coatt_id, p_ids in coatt_p_ids.items():
                            if pid_idx in p_ids:
                                coatt_ids.append(coatt_id)
                                find_coatt_id = True
                                break
                        if not find_coatt_id:
                            coatt_ids.append(0)
                        
                        si = -1
                        # if img_ann['speaking_status'].values in ['speaking', 'vocalizing', 'laughing']:
                            # si = 1
                        # elif img_ann['speaking_status'].values=='not-speaking':
                            # si = 0
                        gt_speaking.append(si)  
                        speaking_scores.append(-1)

                # stack annotations
                coatt_ids = torch.tensor(coatt_ids, dtype=torch.long)
                head_bboxes = torch.stack(head_bboxes)
                gaze_pts = torch.stack(gaze_pts)
                inout = torch.tensor(inout, dtype=torch.float)
                gt_speaking = torch.tensor(gt_speaking, dtype=torch.float)
                speaking_scores = torch.tensor(speaking_scores, dtype=torch.float)
                is_child = torch.tensor(is_child, dtype=torch.float)
                
                #--------------------------------------------------------------------------------
                # Match VLM scores with people
                if self.return_vlm_context:
                    if path in self.loaded_vlm_context.keys():
                        vlm_context = self.loaded_vlm_context[path]
                        p_vlm_context = torch.from_numpy(vlm_context['prompt_score']) # (n, C_person)
                        body_bbox_vlm = torch.from_numpy(vlm_context['bbox_body']) # (n, 4)
                        body_bbox_vlm[:, 2] = body_bbox_vlm[:, 0] + body_bbox_vlm[:, 2]
                        body_bbox_vlm[:, 3] = body_bbox_vlm[:, 1] + body_bbox_vlm[:, 3]
                        dim_vlm = p_vlm_context.shape[1]
                        # Perform matching with vlm head bbox
                        ious_vlm = box_iou(head_bboxes, body_bbox_vlm)
                        max_iou_values, max_iou_indices = torch.max(ious_vlm, dim=1)
                        person_vlm_context = torch.zeros((len(head_bboxes), dim_vlm), dtype=torch.float) - 1
                        for mi, vlm_i in enumerate(max_iou_indices):
                            if max_iou_values[mi]>0.5:
                                person_vlm_context[mi] = p_vlm_context[vlm_i]
                    else:
                        print('VLM features missing: ', path)
                        person_vlm_context = torch.zeros((len(head_bboxes), self.dim_vlm), dtype=torch.float) - 1
                #--------------------------------------------------------------------------------
                
                #--------------------------------------------------------------------------------
                # Match speaking features with people
                if self.return_speaking_features:
                    if path in self.df_speaking_features.groups.keys():
                        frame_speaking_features = self.df_speaking_features.get_group(path)
                        speaking_bboxes = frame_speaking_features[['xmin', 'ymin', 'xmax', 'ymax']].values
                        speaking_bboxes = torch.from_numpy(speaking_bboxes)
                        # Perform matching with head bboxes
                        ious_speaking = box_iou(head_bboxes, speaking_bboxes)
                        max_iou_values, max_iou_indices = torch.max(ious_speaking, dim=1)
                        speaking_features = torch.zeros((len(head_bboxes), 512), dtype=torch.float)
                        for mi, vlm_i in enumerate(max_iou_indices):
                            if max_iou_values[mi]>0.5:
                                speaking_features[mi] = torch.from_numpy(frame_speaking_features['feat'].values[mi][512:])
                    else:
                        print('Speaking features missing: ', path)
                        speaking_features = torch.zeros((len(head_bboxes), 512), dtype=torch.float)
                #--------------------------------------------------------------------------------
                
                # jitter head bboxes
                if self.split == "train":
                    head_bboxes = self.jitter_bbox(head_bboxes, img_w, img_h) 

                # Square head bboxes (can have negative values)
                head_bboxes = square_bbox(head_bboxes, img_w, img_h)

                # Extract Heads
                heads = []
                for head_bbox in head_bboxes:
                    # add to use normalized head bboxes nk
                    head_xmin, head_ymin, head_xmax, head_ymax = head_bbox.tolist()
                    head_xmin, head_xmax = map(lambda x: int(x * img_w), (head_xmin, head_xmax))
                    head_ymin, head_ymax = map(lambda x: int(x * img_h), (head_ymin, head_ymax))
                    head_bbox = torch.tensor([head_xmin, head_ymin, head_xmax, head_ymax], dtype=torch.float32)
                    heads.append(image.crop(head_bbox.int().tolist()))

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
                # gaze_pts[gaze_pts[:, 0] != -1.] /= torch.tensor([img_w, img_h])

                # Normalize Head Bboxes
                # head_bboxes /= torch.tensor([img_w, img_h, img_w, img_h], dtype=float)

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
                        "path": path,
                        "dataset": 'childplay',
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
                    coatt_ids = torch.cat([torch.zeros((num_missing_heads, ), dtype=torch.long), coatt_ids])
                    lah_ids[lah_ids>=0] += num_missing_heads
                    lah_ids = torch.cat([torch.zeros((num_missing_heads, ), dtype=torch.long)-3, lah_ids])
                    speaking_scores = torch.cat([torch.zeros((num_missing_heads, ), dtype=torch.float32)-1, speaking_scores])
                    gt_speaking = torch.cat([torch.zeros((num_missing_heads, ), dtype=torch.float32)-1, gt_speaking])
                    is_child = torch.cat([torch.zeros((num_missing_heads, ), dtype=torch.float32)-1, is_child])
                    if self.return_vlm_context:
                        person_vlm_context = torch.cat([torch.zeros((num_missing_heads, person_vlm_context.shape[1]), 
                                                        dtype=torch.float32)-1, person_vlm_context])
                    if self.return_speaking_features:
                        speaking_features = torch.cat([torch.zeros((num_missing_heads, speaking_features.shape[1]), 
                                                        dtype=torch.float32)-1, speaking_features])

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
                
                if self.return_depth:
                    pcd = sample['pcd'].permute(1,2,0)
                    # compute 3d gaze vectors
                    gaze_pts_img = (gaze_pts* torch.tensor([img_w, img_h])).int()
                    gaze_pts_img[inout!=1] = -1
                    gaze_pts_3d = pcd[gaze_pts_img[:,1]-1, gaze_pts_img[:,0]-1]
                    head_centers = (head_centers* torch.tensor([img_w, img_h])).int()
                    head_centers_3d = pcd[head_centers[:,1],head_centers[:,0]]
                    gaze_vecs_3d = F.normalize(gaze_pts_3d - head_centers_3d, p=2, dim=-1)  # gaze vecs in camera coordinate system
                    dirEyes = F.normalize(head_centers_3d, p=2, dim=-1)
                    cam2eye = CameraToEyeMatrix(dirEyes, inout)  # get camera to eye coordinate matrix
                    gaze_vecs_3d = torch.matmul(cam2eye, gaze_vecs_3d.unsqueeze(-1))  # gaze vecs in eye coordinate system
                    sample['gaze_vecs_3d'] = gaze_vecs_3d.squeeze()
                    sample['cam2eye'] = cam2eye

                laeo_ids = lah2laeo(lah_ids)
                # coatt_ids = lah2coatt(lah_ids)

                # generate gaze heatmaps
                sample["gaze_heatmaps"] = generate_gaze_heatmap(gaze_pts, sigma=self.heatmap_sigma, size=self.heatmap_size)
                coatt_heatmaps, coatt_levels = generate_coatt_heatmap(sample["gaze_heatmaps"], coatt_ids, self.num_coatt, size=self.heatmap_size)
                sample["coatt_heatmaps"] = coatt_heatmaps
                sample["coatt_levels"] = coatt_levels

                sample['lah_ids'] = lah_ids
                sample['laeo_ids'] = laeo_ids
                sample['coatt_ids'] = coatt_ids
                sample['head_bboxes'] = head_bboxes
                sample['gaze_pts'] = gaze_pts
                sample['heads'] = heads
                sample['inout'] = inout
        #         mask = gt_speaking!=-1
        #         speaking_scores = gt_speaking * mask.int() + speaking_scores * (1 - mask.int())
                sample['speaking'] = gt_speaking    # speaking_scores, gt_speaking
                
                # Append current frame annotations to temporal sample
                t_sample['image'].append(sample['image'])
                t_sample['heads'].append(sample['heads'])
                t_sample['head_centers'].append(sample['head_centers'])
                t_sample['head_masks'].append(sample['head_masks'])
                t_sample['head_bboxes'].append(sample['head_bboxes'])
                t_sample['gaze_pts'].append(sample['gaze_pts'])
                t_sample['gaze_vecs'].append(sample['gaze_vecs'])
                t_sample['gaze_heatmaps'].append(sample['gaze_heatmaps'])
                t_sample['coatt_heatmaps'].append(sample['coatt_heatmaps'])
                t_sample['coatt_levels'].append(sample['coatt_levels'])
                t_sample['inout'].append(sample['inout'])
                t_sample['lah_ids'].append(sample['lah_ids'])
                t_sample['laeo_ids'].append(sample['laeo_ids'])
                t_sample['coatt_ids'].append(sample['coatt_ids'])
                t_sample['speaking'].append(sample['speaking'])
                t_sample['is_child'].append(is_child)
                t_sample['num_valid_people'].append(sample['num_valid_people'])
                t_sample['img_size'].append(sample['img_size'])
                t_sample['path'].append(path)
                if self.return_vlm_context:
                    t_sample['person_vlm_context'].append(person_vlm_context)
                if self.return_depth:
                    t_sample['pcd'].append(sample['pcd'])
                    t_sample['gaze_vecs_3d'].append(sample['gaze_vecs_3d'])
                    t_sample['cam2eye'].append(sample['cam2eye'])
                if self.return_speaking_features:
                    t_sample['speaking_features'].append(speaking_features)

        for key, item in t_sample.items():
            if key not in ['dataset', 'path', 'pids']:
                t_sample[key] = torch.stack(t_sample[key], axis=0).squeeze()
                if self.temporal_context==0:
                    t_sample[key] = t_sample[key].unsqueeze(0)
        return t_sample

    def __len__(self):
        # return len(self.keys)
        # return len(self.paths)

        # self.use_ratio = 0.05
        # self.use_ratio = 0.1
        self.use_ratio = 1.0
        return int(len(self.paths) * self.use_ratio)

# ============================================================================================================ #
#                                              CHILDPLAY DATA MODULE                                          #
# ============================================================================================================ #
#TODO: update mean/std normalization from GazeFollow to ChildPlay values
IMG_MEAN = [0.44232, 0.40506, 0.36457]
IMG_STD = [0.28674, 0.27776, 0.27995]
class ChildPlayDataModule(pl.LightningDataModule):
    def __init__(
        self,
        cfg,
        root: str,
        root_depth,
        root_focal,
        batch_size: Union[int, dict] = 32,
        num_people: int = 5,
        temporal_context: int=2,
        temporal_stride: int=3,
        image_size = (224,224),
        heatmap_size = 64,
        return_depth: bool=False,
        dim_vlm = 117
    ):  
        
        super().__init__()
        self.cfg = cfg
        self.root = root
        self.root_depth = root_depth
        self.root_focal = root_focal
        self.num_people = num_people
        if type(image_size)==ListConfig:
            image_size = OmegaConf.to_object(image_size)
        self.image_size = pair(image_size)
        self.heatmap_sigma = int(np.mean(heatmap_size)*3/64)
        self.heatmap_size = heatmap_size
        self.batch_size = (
            {stage: batch_size for stage in Stage}
            if isinstance(batch_size, int)
            else batch_size
        )
        self.temporal_context=temporal_context
        self.temporal_stride=temporal_stride
        self.return_depth=return_depth
        self.dim_vlm = dim_vlm


    def setup(self, stage: str):
        if stage == "fit":
            train_transform = Compose(
                [
                    RandomCropSafeGaze(aspect=1.0, p=1.0),
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
                        img_mean=IMG_MEAN,
                        img_std=IMG_STD
                    ),
                ]
            )
            self.train_dataset = ChildPlayDataset_temporal(
                cfg=self.cfg,
                root=self.root,
                root_depth=self.root_depth,
                root_focal=self.root_focal, 
                split="train", 
                stride=12,
                transform=train_transform, 
                tr=(-0.1, 0.1), 
                num_people=self.num_people["train"],
                temporal_context=self.temporal_context,
                temporal_stride=self.temporal_stride,
                heatmap_size=self.heatmap_size,
                return_depth=self.return_depth,
                dim_vlm=self.dim_vlm
            )

            val_transform = Compose(
                [
                    Resize(self.image_size, head_size=(224, 224)),
                    ToTensor(),
                    Normalize(
                        img_mean=IMG_MEAN,
                        img_std=IMG_STD
                    ),
                ]
            )
            self.val_dataset = ChildPlayDataset_temporal(
                cfg=self.cfg,
                root=self.root,
                root_depth=self.root_depth,
                root_focal=self.root_focal, 
                split="val", 
                stride=6,
                transform=val_transform, 
                tr=(0.0, 0.0), 
                num_people=self.num_people["val"],
                temporal_context=self.temporal_context,
                temporal_stride=self.temporal_stride,
                heatmap_size=self.heatmap_size,
                return_depth=self.return_depth,
                dim_vlm=self.dim_vlm
            )

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
            self.val_dataset = ChildPlayDataset_temporal(
                cfg=self.cfg,
                root=self.root,
                root_depth=self.root_depth,
                root_focal=self.root_focal, 
                split="val",
                stride=6,
                transform=val_transform, 
                tr=(0.0, 0.0), 
                num_people=self.num_people["val"],
                temporal_context=self.temporal_context,
                temporal_stride=self.temporal_stride,
                heatmap_size=self.heatmap_size,
                return_depth=self.return_depth,
                dim_vlm=self.dim_vlm
            )

        elif stage == "test":
            test_transform = Compose(
                [
                    Resize(img_size=self.image_size, head_size=(224, 224)),
                    ToTensor(),
                    Normalize(
                        img_mean=IMG_MEAN,
                        img_std=IMG_STD
                    ),
                ]
            )
            self.test_dataset = ChildPlayDataset_temporal(
                cfg=self.cfg,
                root=self.root,
                root_depth=self.root_depth,
                root_focal=self.root_focal, 
                split="test", 
                stride=3,
                transform=test_transform, 
                tr=(0.0, 0.0), 
                num_people=self.num_people["test"],
                temporal_context=self.temporal_context,
                temporal_stride=self.temporal_stride,
                heatmap_size=self.heatmap_size,
                return_depth=self.return_depth,
                dim_vlm=self.dim_vlm
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
            self.predict_dataset = ChildPlayDataset_temporal(
                cfg=self.cfg,
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
            shuffle=False,
            num_workers=4,
            pin_memory=False,
        )
        return dataloader

    def predict_dataloader(self):
        dataloader = DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size[Stage.PREDICT],
            shuffle=False,
            num_workers=4,
            pin_memory=False,
        )
        return dataloader
