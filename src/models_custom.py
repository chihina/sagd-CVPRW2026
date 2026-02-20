import math
import pickle
from collections import OrderedDict
from typing import Dict, Union
import itertools
import sys
import cv2
from PIL import Image
import os
import json
import matplotlib.pyplot as plt
import time

import networkx as nx
import community as community

import numpy as np
import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchmetrics as tm
import wandb
from termcolor import colored
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, CosineAnnealingWarmRestarts, StepLR, MultiStepLR
from timm.scheduler import CosineLRScheduler
# from torch_linear_assignment import batch_linear_assignment, assignment_to_indices

from src.losses import compute_chong_loss, compute_sharingan_loss, compute_rinnegan_loss, compute_social_loss, compute_interact_loss, compute_inout_loss, compute_coatt_loss, compute_gazelle_loss
from src.metrics import AUC, Distance, GFTestAUC, GFTestDistance, GroupCost, GroupAP, GroupAPFast
from src.networks.chong import ChongNet, GazeBaseline, NoraNet
# from src.networks.rinnegan import Rinnegan
# from src.networks.rinnegan_multivit import MultiViTRinnegan
from src.networks.sharingan import Sharingan
from src.networks.sharingan_social import Sharingan_social
from src.networks.interact_net_temporal import InteractNet
from src.networks.geom_gaze import GeomGaze
from src.utils import spatial_argmax2d
# from src.utils import id_to_pairwise_coatt, id_to_pairwise_lah, id_to_pairwise_laeo
# from src.utils import id_to_pairwise_coatt_vectorized, id_to_pairwise_laeo_vectorized, id_to_pairwise_lah_vectorized
from src.utils import id_to_pairwise_coatt_vectorized as id_to_pairwise_coatt
from src.utils import id_to_pairwise_laeo_vectorized as id_to_pairwise_laeo
from src.utils import id_to_pairwise_lah_vectorized as id_to_pairwise_lah

TERM_COLOR = "cyan"


# ==================================================================================================================
#                                                 INTERACT MODEL                                                  #
# ==================================================================================================================
class GazeLLE(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        
        print('---------------------------------------------------------')
        print('Experiment: ', cfg.experiment.name)
        print('---------------------------------------------------------')

        self.model_name = cfg.model.model_name
        # self.model, transform = torch.hub.load('fkryan/gazelle', 'gazelle_dinov2_vitb14')
        self.model, transform = torch.hub.load('fkryan/gazelle', 'gazelle_dinov2_vitb14_inout')

        self.cfg = cfg
        self.test_step_outputs = []
        self.metrics = nn.ModuleDict({"val_dist": Distance(), "test_dist": Distance(), "test_auc": AUC()})

        self.num_tranining_samples = cfg.data.num_samples  # TODO: update this. VAT: ?, GazeFollow: 108955
        self.num_steps_in_epoch = math.ceil(self.num_tranining_samples / cfg.train.batch_size)
        self.compute_hm_loss = compute_gazelle_loss

        print(colored(f"Freezing the Backbone layers.", TERM_COLOR))
        self.freeze_module(self.model.backbone)

    def _init_weights(self):
        # Load pre-trained weights
        if self.model_weights:
            model_ckpt = torch.load(self.model_weights, map_location="cpu")
            model_weights = OrderedDict(
                [
                    (name.replace("model.", ""), value)
                    for name, value in model_ckpt["state_dict"].items()
                ]
            )
            if self.model_name=='sharingan_social':
                model_weights = OrderedDict(
                    [
                        (name.replace("decoder.", "gaze_decoder."), value)
                        for name, value in model_weights.items()
                    ]
                )
                model_weights = OrderedDict(
                    [
                        (name.replace("encoder.encoder", "encoder.blocks"), value)
                        for name, value in model_weights.items()
                    ]
                )
            self.model.load_state_dict(model_weights, strict=False)
            print(
                colored(
                    f"Successfully loaded pre-trained weights from {self.model_weights}.",
                    TERM_COLOR,
                )
            )
            del model_ckpt
        else:
            # Load weights for Multi ViT
            if self.model_name in ['sharingan_social', 'gaze_interact'] and self.multivit_weights:
                multivit_ckpt = torch.load(self.multivit_weights, map_location="cpu")
                image_tokenizer_weights = OrderedDict([(name.replace("input_adapters.rgb.", ""), value) for name, value in multivit_ckpt["model"].items() if "input_adapters.rgb" in name])
                self.model.image_tokenizer.load_state_dict(image_tokenizer_weights, strict=True)
                print(colored(f"Successfully loaded weights for the image tokenizer from {self.multivit_weights}.", TERM_COLOR))

                encoder_weights = OrderedDict([(name.replace("encoder.", ""), value) for name, value in multivit_ckpt["model"].items() if "encoder" in name])
                self.model.encoder.blocks.load_state_dict(encoder_weights, strict=True)
                print(colored(f"Successfully loaded weights for the ViT encoder from {self.multivit_weights}.", TERM_COLOR))
                
                del multivit_ckpt, image_tokenizer_weights, encoder_weights

            # Load Gaze Encoder Gaze360 Pre-trained Weights
            # gaze360_ckpt = torch.load(self.gaze_weights, map_location="cpu")
            # gaze360_weights = OrderedDict([(name.replace("base_head.", ""), value) for name, value in gaze360_ckpt["model_state_dict"].items() if "base_head" in name])
            # self.model.gaze_encoder.backbone.load_state_dict(gaze360_weights, strict=True)
            # print(colored(f"Successfully loaded weights for the gaze backbone from {self.gaze_weights}.", TERM_COLOR))

            # Delete checkpoints
            # del gaze360_ckpt, gaze360_weights

    def _set_batchnorm_eval(self, model):
        for module in model.modules():
                module.eval()

    def freeze_module(self, module):
        for param in module.parameters():
            param.requires_grad = False

    # def _freeze(self):
        # if self.cfg.train.freeze.gaze_encoder_backbone:
        #     print(colored(f"Freezing the Gaze Encoder backbone layers.", TERM_COLOR))
        #     self.freeze_module(self.model.gaze_encoder.backbone)
        # if self.cfg.train.freeze.gaze_encoder:
        #     print(colored(f"Freezing the Gaze Encoder layers.", TERM_COLOR))
        #     self.freeze_module(self.model.gaze_encoder)
        # if self.cfg.train.freeze.image_tokenizer:
        #     print(colored(f"Freezing the Image Tokenizer layers.", TERM_COLOR))
        #     self.freeze_module(self.model.image_tokenizer)
        # if self.cfg.train.freeze.vit_encoder:
        #     print(colored(f"Freezing the ViT Encoder layers.", TERM_COLOR))
        #     self.freeze_module(self.model.encoder)
        # if self.cfg.train.freeze.vit_adaptor:
        #     print(colored(f"Freezing the ViT Adaptor layers.", TERM_COLOR))
        #     self.freeze_module(self.model.vit_adaptor)

        # if self.cfg.train.freeze.gaze_decoder:
            # print(colored(f"Freezing the Gaze Decoder layers.", TERM_COLOR))
            # self.freeze_module(self.model.gaze_decoder)
        # if self.cfg.train.freeze.gaze_decoder:
            # print(colored(f"Freezing the Gaze Heatmap Decoder layers.", TERM_COLOR))
            # self.freeze_module(self.model.gaze_hm_decoder_new)

        # if self.cfg.train.freeze.inout_decoder:
            # print(colored(f"Freezing the InOut Decoder layers.", TERM_COLOR))
            # self.freeze_module(self.model.inout_decoder)


    def forward(self, batch):
        # update batch for single frame input
        batch['images'] = batch['images'][:,0,:,:,:]  # B, T, C, H, W -> B, C, H, W
        
        batch['bboxes'] = batch['bboxes'][:,0,:,:]    # B, T, N, 4 -> B, N, 4
        batch['bboxes'] = batch['bboxes'].cpu().numpy().tolist()

        return self.model(batch)

    def get_input_head_maps(self, bboxes):
        # bboxes: [[(xmin, ymin, xmax, ymax)]] - list of list of head bboxes per image
        head_maps = []
        for bbox_list in bboxes:
            img_head_maps = []
            for bbox in bbox_list:
                if bbox is None: # no bbox provided, use empty head map
                    img_head_maps.append(torch.zeros(self.featmap_h, self.featmap_w))
                else:
                    xmin, ymin, xmax, ymax = bbox
                    width, height = self.featmap_w, self.featmap_h
                    print(xmin, ymin, xmax, ymax, width, height)
                    assert False, 'stop'
                    xmin = round(xmin * width)
                    ymin = round(ymin * height)
                    xmax = round(xmax * width)
                    ymax = round(ymax * height)
                    head_map = torch.zeros((height, width))
                    head_map[ymin:ymax, xmin:xmax] = 1
                    img_head_maps.append(head_map)
            head_maps.append(torch.stack(img_head_maps))
        return head_maps

    def configure_optimizers(self):
        
        # separate params for temporal modelling
        if self.model_name=='gaze_interact':
            temporal_params = [{"params": self.model.gaze_encoder_temporal.parameters(), 
                                "name": "gaze-encoder-temporal", 
                                "lr": self.cfg.optimizer.lr*3,
                                "init_lr": self.cfg.optimizer.lr*3}, 

                               {"params": self.model.people_temporal.parameters(), 
                                "name": "people-temporal", 
                                "lr": self.cfg.optimizer.lr*3,
                                "init_lr": self.cfg.optimizer.lr*3},
                               
                               {"params": self.model.decoder_sa.parameters(), 
                                "name": "decoder-sa", 
                                "lr": self.cfg.optimizer.lr*3,
                                "init_lr": self.cfg.optimizer.lr*3},

                                {"params": self.model.decoder_lah.parameters(), 
                                    "name": "decoder-lah", 
                                    "lr": self.cfg.optimizer.lr*3,
                                    "init_lr": self.cfg.optimizer.lr*3},
                                ]
        else:
            temporal_params = []
        
        other_params = []
        for k,v in self.model.named_parameters():
            if ('_temporal' not in k) and ('decoder_sa' not in k) and ('decoder_lah' not in k):
                other_params.append(v)
        other_params = [{"params": other_params,
                         "name": "base", 
                         "lr": self.cfg.optimizer.lr,
                         "init_lr": self.cfg.optimizer.lr}]
        
        params = temporal_params + other_params
        optimizer = optim.AdamW(params, weight_decay=self.cfg.optimizer.weight_decay)

        # cosine annealing
        if self.cfg.scheduler.type=='CosineAnnealingWarmRestarts':
            T_0 = self.cfg.scheduler.t_0_epochs * self.num_steps_in_epoch
            T_mult = self.cfg.scheduler.t_mult
            lr_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0, T_mult=T_mult, eta_min=0)
            lr_scheduler_config = {"scheduler": lr_scheduler, "interval": 'step', "frequency": 1}
        elif self.cfg.scheduler.type=='StepLR':
            # lr_scheduler = MultiStepLR(optimizer, milestones=[10,11,12,13,14,15,16], gamma=0.5)
            # lr_scheduler = StepLR(optimizer, step_size=self.cfg.scheduler.t_0_epochs, gamma=0.1)
            lr_scheduler = StepLR(optimizer, step_size=self.cfg.scheduler.t_0_epochs, gamma=1)
            lr_scheduler_config = {"scheduler": lr_scheduler, "interval": "epoch", "frequency": 1}
        else:
            print('Invalid scheduler selected...')
        
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}

    def lr_scheduler_step(self, scheduler, *args, **kwargs):
        # Step scheduler
        scheduler.step()

        # Warm-up Steps
        n = self.cfg.scheduler.warmup_epochs * self.num_steps_in_epoch
        if self.trainer.global_step < n:
            lr_scale = min(1.0, float(self.trainer.global_step + 1) / n)
            # optimizer
            for pg in scheduler.optimizer.param_groups:
                pg["lr"] = lr_scale * pg["init_lr"]

    def on_train_epoch_start(self):
        if self.current_epoch == self.trainer.max_epochs - 1:
            # Workaround to always save the last epoch until the bug is fixed in lightning (https://github.com/Lightning-AI/lightning/issues/4539)
            self.trainer.check_val_every_n_epoch = 1

            # Disable backward pass for SWA until the bug is fixed in lightning (https://github.com/Lightning-AI/lightning/issues/17245)
            self.automatic_optimization = False

        # Set BN layers to eval mode for frozen modules
        # if self.cfg.train.freeze.gaze_encoder:
            # self.model.gaze_encoder.apply(self._set_batchnorm_eval)
        # if self.cfg.train.freeze.image_tokenizer:
            # self.model.image_tokenizer.apply(self._set_batchnorm_eval)
        # if self.cfg.train.freeze.vit_encoder:
            # self.model.encoder.apply(self._set_batchnorm_eval)

        # if self.cfg.train.freeze.gaze_decoder:
            # self.model.gaze_decoder.apply(self._set_batchnorm_eval)
        # if self.cfg.train.freeze.gaze_decoder:
            # self.model.gaze_hm_decoder_new.apply(self._set_batchnorm_eval)

        # if self.cfg.train.freeze.inout_decoder:
            # self.model.inout_decoder.apply(self._set_batchnorm_eval)

    def training_step(self, batch, batch_idx):
        nv = int((batch["speaking"]!=-1).sum().item())
        ni = int((batch["inout"]==1).sum().item())

        gaze_hm_gt = batch["gaze_heatmaps"]  # B, T, N, H, W
        batch_size, t, n, hm_h, hm_w = gaze_hm_gt.shape
        inout_gt = batch["inout"]  # B, T, N
        gaze_hm_gt = gaze_hm_gt.view(batch_size*t, n, hm_h, hm_w)
        inout_gt = inout_gt.view(batch_size*t, -1)

        # Forward pass
        out = self(batch)

        # get gaze heatmap and inout predictions
        gaze_hm_pred = [i for i in out['heatmap']]
        gaze_hm_pred = torch.stack(gaze_hm_pred)
        inout_pred = [i for i in out['inout']]
        inout_pred = torch.stack(inout_pred)

        loss_dist, logs_dist = self.compute_hm_loss(gaze_hm_gt, inout_gt, gaze_hm_pred, inout_pred)    # 3d gaze vector loss
        loss = loss_dist

        # Logging Distance, InOut losses            
        self.log("loss/train/heatmap", logs_dist["heatmap_loss"], batch_size=ni, prog_bar=False, on_step=True, on_epoch=True)
        self.log("loss/train/inout", logs_dist["inout_loss"], batch_size=n, prog_bar=False, on_step=True, on_epoch=True)
        self.log("loss/train", loss.item(), batch_size=n, prog_bar=True, on_step=True, on_epoch=True)
        
        # Logging metrics
        self.log("metric/train/dist", loss.item(), batch_size=n, prog_bar=False, on_step=True, on_epoch=True)

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        # nv = int((batch["speaking"]!=-1).sum().item())
        ni = int((batch["inout"]==1).sum().item())

        gaze_hm_gt = batch["gaze_heatmaps"]  # B, T, N, H, W
        batch_size, t, n, hm_h, hm_w = gaze_hm_gt.shape
        inout_gt = batch["inout"]  # B, T, N
        gaze_hm_gt = gaze_hm_gt.view(batch_size*t, n, hm_h, hm_w)
        inout_gt = inout_gt.view(batch_size*t, -1)

        # Forward pass
        out = self(batch)

        # get gaze heatmap and inout predictions
        gaze_hm_pred = [i for i in out['heatmap']]
        gaze_hm_pred = torch.stack(gaze_hm_pred)
        inout_pred = [i for i in out['inout']]
        inout_pred = torch.stack(inout_pred)

        loss_dist, logs_dist = self.compute_hm_loss(gaze_hm_gt, inout_gt, gaze_hm_pred, inout_pred)    # 3d gaze vector loss
        loss = loss_dist

        # Logging losses
        self.log("loss/val/heatmap", logs_dist["heatmap_loss"], batch_size=ni, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        self.log("loss/val/inout", logs_dist["inout_loss"], batch_size=n, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        self.log("loss/val", loss.item(), batch_size=n, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        
        # Logging metrics
        self.log("metric/val/dist", loss.item(), batch_size=n, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        ni = int((batch["inout"]==1).sum().item())
        # assert n == ni, f"Expected all test samples to be looking inside. Got {n} samples, {ni} of which are looking inside."

        gaze_hm_gt = batch["gaze_heatmaps"]  # B, T, N, H, W
        batch_size, t, n, hm_h, hm_w = gaze_hm_gt.shape
        inout_gt = batch["inout"]  # B, T, N
        gaze_hm_gt = gaze_hm_gt.view(batch_size*t, n, hm_h, hm_w)
        inout_gt = inout_gt.view(batch_size*t, -1)

        # Forward pass
        out = self(batch)

        # get gaze heatmap and inout predictions
        gaze_hm_pred = [i for i in out['heatmap']]
        gaze_hm_pred = torch.stack(gaze_hm_pred)  # B*T, num_people, H, W
        inout_pred = [i for i in out['inout']]
        inout_pred = torch.stack(inout_pred)  # B*T, num_people
        head_bboxes = batch['head_bboxes']  # B, T, N, 4

        gaze_pt_pred = []
        gaze_vec_pred = []
        for b_idx in range(gaze_hm_pred.shape[0]):
            hm = gaze_hm_pred[b_idx]
            num_people, _, _ = hm.shape
            gp = spatial_argmax2d(hm, normalize=True).view(num_people, 2)
            gaze_pt_pred.append(gp)
            heads_b = head_bboxes[b_idx // t, b_idx % t, :, :]  # N, 4
            gv_p = []
            for p_idx in range(num_people):
                head_bbox = heads_b[p_idx]
                head_cx = (head_bbox[0] + head_bbox[2]) / 2.0
                head_cy = (head_bbox[1] + head_bbox[3]) / 2.0
                gaze_cx = gp[p_idx, 0]
                gaze_cy = gp[p_idx, 1]
                gv_p.append(torch.tensor([gaze_cx - head_cx, gaze_cy - head_cy], device=gp.device))
            gv_p = torch.stack(gv_p, dim=0)
            gaze_vec_pred.append(gv_p)
        gaze_pt_pred = torch.stack(gaze_pt_pred)
        gaze_vec_pred = torch.stack(gaze_vec_pred)

        # only take outputs of central frame
        t, num_people, hm_h, hm_w = gaze_hm_pred.shape
        middle_frame_idx = int(t/2)
        coatt_levels_gt = batch['coatt_levels'][:,middle_frame_idx,:,:]
        coatt_hm_gt = batch['coatt_heatmaps'][:,middle_frame_idx,:,:,:]

        # >>>>>>>>> grouping metrics based on post-processing <<<<<<<<<<
        inout_thr = 0.5
        dist_thr_list = np.linspace(0, 0.2, 5).tolist()
        dist_thr_list = [round(d, 2) for d in dist_thr_list]

        # fill zero if the head is padded (looking outside)
        head_bboxes = batch['head_bboxes'][:,middle_frame_idx,:,:]
        head_pad_mask = torch.sum(head_bboxes, dim=-1) <= 0
        inout_pred[head_pad_mask] = 0

        # generate inout mask based on inout prediction
        out_mask = inout_pred[0] < inout_thr

        metrics_pp = {}
        coatt_level_pred_grp_all_pp = {}
        coatt_hm_pred_grp_all_pp = {}
        for dist_thr in dist_thr_list:
            coatt_level_pred_label_pp = []
            grp_pis = set()
            for pi in range(num_people):
                if pi in grp_pis:
                    continue
                gaze_pt_pred_p = gaze_pt_pred[0, pi]
                dists_xy = gaze_pt_pred[0] - gaze_pt_pred_p
                dists = torch.norm(dists_xy, dim=-1)
                # dists[out_mask] = float('inf')
                close_indices = torch.where((dists<dist_thr).int())[0]
                if len(close_indices)>1:
                    coatt_level_pred_grp = torch.zeros((num_people), device=gaze_pt_pred.device)
                    coatt_level_pred_grp[close_indices] = 1
                    coatt_level_pred_label_pp.append(coatt_level_pred_grp)
                    grp_pis = grp_pis | set(close_indices.tolist())
            
            if len(coatt_level_pred_label_pp)==0:
                coatt_level_pred_label_pp = torch.zeros((1, 1, num_people), device=gaze_pt_pred.device)
            else:
                coatt_level_pred_label_pp = torch.stack(coatt_level_pred_label_pp, dim=0).unsqueeze(0)
            
            group_cost = GroupCost()
            coatt_hm_pred_pp_sample = self.generate_coatt_hm_from_level(coatt_level_pred_label_pp, gaze_hm_pred)
            ret_metrics_pp = group_cost.compute(coatt_level_pred_label_pp, coatt_levels_gt, coatt_hm_pred_pp_sample, coatt_hm_gt)
            metrics_pp[dist_thr] = [ret_metrics_pp['Group_IoU'], ret_metrics_pp['Group_Dist']]
            self.log(f"metric/test/coatt_cost_iou_pp_{dist_thr}", ret_metrics_pp['Group_IoU'], batch_size=1, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
            self.log(f"metric/test/coatt_cost_dist_pp_{dist_thr}", ret_metrics_pp['Group_Dist'], batch_size=1, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
            coatt_level_pred_grp_all_pp[dist_thr] = coatt_level_pred_label_pp
            coatt_hm_pred_grp_all_pp[dist_thr] = coatt_hm_pred_pp_sample

        '''
        # visualize test results for the middle frame
        img = batch['images'][0, :, :, :].cpu().numpy()  # C, H, W
        img = np.transpose(img, (1,2,0))  # H, W, C
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img = img * 255.0
        img = img.astype(np.uint8)
        img = np.ascontiguousarray(img)
        head_bboxes = batch['head_bboxes'][0, middle_frame_idx, :, :].cpu().numpy()  # N, 4
        gaze_pts = batch['gaze_pts'][0, middle_frame_idx, :, :].cpu().numpy()  # N, 2
        gaze_pts_pred_np = gaze_pt_pred[0].cpu().numpy()  # N, 2
        for pi in range(head_bboxes.shape[0]):
            xmin, ymin, xmax, ymax = head_bboxes[pi]
            print(f'Person {pi}: Head bbox: ({xmin}, {ymin}, {xmax}, {ymax}), GT gaze pt: ({gaze_pts[pi][0]}, {gaze_pts[pi][1]}), Pred gaze pt: ({gaze_pts_pred_np[pi][0]}, {gaze_pts_pred_np[pi][1]})')
            if xmin<=0 and ymin<=0 or xmax<=0 or ymax<=0:
                continue
            # draw head bbox
            cv2.rectangle(img, (int(xmin*img.shape[1]), int(ymin*img.shape[0])), (int(xmax*img.shape[1]), int(ymax*img.shape[0])), (0,255,0), 2)
            # draw gt gaze point
            # gp_x, gp_y = gaze_pts[pi]
            # if gp_x < 0 or gp_y < 0:
                # pass
            # else:
                # cv2.circle(img, (int(gp_x*img.shape[1]), int(gp_y*img.shape[0])), 5, (255,0,0), -1)
            # draw pred gaze point
            gp_x_p, gp_y_p = gaze_pts_pred_np[pi]
            if gp_x_p < 0 or gp_y_p < 0:
                pass
            else:
                cv2.circle(img, (int(gp_x_p*img.shape[1]), int(gp_y_p*img.shape[0])), 5, (0,0,255), -1)

            # draw arrows from head center to gaze point
            head_cx = (xmin + xmax) / 2.0
            head_cy = (ymin + ymax) / 2.0
            if gp_x_p >= 0 and gp_y_p >= 0:
                cv2.arrowedLine(img, (int(head_cx*img.shape[1]), int(head_cy*img.shape[0])), (int(gp_x_p*img.shape[1]), int(gp_y_p*img.shape[0])), (255,255,0), 2, tipLength=0.2)

        vis_img = img
        cv2.imwrite(os.path.join(f'test_vis_{batch_idx}.png'), vis_img)
        assert False, 'stop'
        '''

        # post-processing
        best_res_pp = max(metrics_pp, key=lambda x: metrics_pp[x][0])
        metrics_pp_best = metrics_pp[best_res_pp]
        coatt_level_pred_label_pp_best = coatt_level_pred_grp_all_pp[best_res_pp]
        coatt_hm_pred_pp_best = coatt_hm_pred_grp_all_pp[best_res_pp]
        coatt_hm_pred_pp_pk_cd = spatial_argmax2d(coatt_hm_pred_pp_best.reshape(-1, coatt_hm_pred_pp_best.shape[2], coatt_hm_pred_pp_best.shape[3]), normalize=True).view(batch_size, -1, 2)

        # obtain coatt_hm peak values
        coatt_hm_gt_pk_cd = spatial_argmax2d(coatt_hm_gt.reshape(-1, coatt_hm_gt.shape[2], coatt_hm_gt.shape[3]), normalize=True).view(batch_size, -1, 2)
        coatt_hm_gt_pk_val = coatt_hm_gt.reshape(-1, coatt_hm_gt.shape[2]*coatt_hm_gt.shape[3]).max(dim=-1).values.view(batch_size, -1)

        # Build output dict
        output = {"head_bboxes": batch["head_bboxes"][:,middle_frame_idx,:, :],
                  "gp_pred": gaze_pt_pred, 
                  "gp_gt": batch["gaze_pts"][:,middle_frame_idx,:, :],
                  "gv_pred": gaze_vec_pred, 
                  "gv_gt": batch["gaze_vecs"][:,middle_frame_idx,:, :],
                  "gaze_pts_pred" : gaze_pt_pred,
                  "gaze_pts_gt" : batch["gaze_pts"][:,middle_frame_idx,:, :],
                  'gaze_pts_pred': gaze_pt_pred,
                #   # optionally save gaze heatmaps
                  "hm_pred": gaze_hm_pred,
                #   "hm_gt": batch["gaze_heatmaps"][:,middle_frame_idx,:,:,:],
                  "coatt_hm_pred_pp_pk_cd": coatt_hm_pred_pp_pk_cd,
                  "coatt_hm_gt_pk_cd": coatt_hm_gt_pk_cd,
                  "coatt_hm_gt_pk_val": coatt_hm_gt_pk_val,
                  "inout_gt": inout_gt, 
                  "path": batch["path"],
                  "inout_pred": inout_pred,
                #   "coatt_pred": coatt_pred,
                #   "coatt_hm_pred": coatt_hm_pred_pp_sample,
                #   "coatt_hm_gt": coatt_hm_gt,
                  "coatt_level_pred": coatt_level_pred_label_pp,
                #   "coatt_level_pred_label_pairs": coatt_level_pred_label_pairs,
                  "coatt_level_pred_grp_best_pp": coatt_level_pred_label_pp_best,
                  "coatt_level_gt": batch['coatt_levels'][:,middle_frame_idx,:,:],
                #   "laeo_pred": laeo_pred,
                #   "lah_pred": lah_pred,
                #   "coatt_gt": coatt_gt,
                #   "laeo_gt": laeo_gt,
                #   "laeo_ids": batch['laeo_ids'][:,middle_frame_idx,:],
                #   "lah_gt": lah_gt,
                  "dataset": batch['dataset'],
                  "is_child": batch['is_child'][:,middle_frame_idx,:],
                  "speaking": batch['speaking'][:,middle_frame_idx,:],
                  "num_valid_people": batch['num_valid_people'],
                  "group_iou_pp": metrics_pp_best[0],
                  "group_dist_pp": metrics_pp_best[1],
                  "best_dist_pp": best_res_pp,
                  "path": batch["path"],
                  }
        self.test_step_outputs.append(output)

    def edge_base_grp_detection(self, coatt_pred: torch.Tensor, coatt_level_gt: torch.Tensor, inout_pred: torch.Tensor, res: float) -> torch.Tensor:
        num_people = coatt_level_gt.shape[-1]
        indices = torch.tensor(list(itertools.permutations(torch.arange(num_people), 2)))

        edges = []
        for ind in range(indices.shape[0]):
            i, j = indices[ind]
            i_inout_pred, j_inout_pred = inout_pred[0, i], inout_pred[0, j]
            weight = coatt_pred[0, ind].item()
            edges.append((i.item(), j.item(), {'weight': weight}))
        G = nx.Graph()
        G.add_edges_from(edges)
        partition = community.best_partition(G, weight='weight', resolution=res)
        group_ids = list(set(partition.values()))

        coatt_level_pred = []
        for g_id in group_ids:
            members = [node for node, comm_id in partition.items() if comm_id == g_id]
            if len(members)>1:
                coatt_level_pred_grp = torch.zeros(num_people, device=coatt_level_gt.device)
                coatt_level_pred_grp[members] = 1
                coatt_level_pred.append(coatt_level_pred_grp)

        if len(coatt_level_pred)==0:
            coatt_level_pred = torch.zeros((1, num_people), device=coatt_level_gt.device)
        else:
            coatt_level_pred = torch.stack(coatt_level_pred, dim=0)

        return coatt_level_pred

    def post_process_coatt_pred(self, gaze_pts_pred: list, inout_pred: list, dist_thr: float) -> list:
        coatt_level_pred = []
        inout_thr = 0.5
        for idx in range(len(gaze_pts_pred)):
            gaze_pt_pred = gaze_pts_pred[idx][0]
            inout_pred_b = inout_pred[idx][0]
            out_mask = inout_pred_b < inout_thr

            grp_pis = set()
            num_people = gaze_pt_pred.shape[0]
            coatt_level_pred_sample = []
            for pi in range(num_people):
                if pi in grp_pis:
                    continue
                gaze_pt_pred_p = gaze_pt_pred[pi]
                dists_xy = gaze_pt_pred - gaze_pt_pred_p
                dists = torch.norm(dists_xy, dim=-1)
                dists[out_mask] = float('inf')
                close_indices = torch.where((dists<dist_thr).int())[0]
                if len(close_indices)>1:
                    coatt_level_pred_grp = torch.zeros((num_people), device=gaze_pt_pred.device)
                    coatt_level_pred_grp[close_indices] = 1
                    coatt_level_pred_sample.append(coatt_level_pred_grp)
                    grp_pis = grp_pis | set(close_indices.tolist())
            
            if len(coatt_level_pred_sample)==0:
                coatt_level_pred_sample = torch.zeros((1, 1, num_people), device=gaze_pt_pred.device)
            else:
                coatt_level_pred_sample = torch.stack(coatt_level_pred_sample, dim=0).unsqueeze(0)
            
            coatt_level_pred.append(coatt_level_pred_sample)

        return coatt_level_pred
    
    # combine heatmaps of each person attention based on group prediction
    def generate_coatt_hm_from_level(self, coatt_level_pred: torch.Tensor, hm_pred: torch.Tensor) -> torch.Tensor:
        people_num = hm_pred.shape[1]
        img_h, img_w = hm_pred.shape[2], hm_pred.shape[3]
        coatt_level_pred = coatt_level_pred.view(-1, people_num, 1, 1)
        coatt_hm_pred = torch.sum(hm_pred * coatt_level_pred, dim=1)
        coatt_hm_pred = coatt_hm_pred.view(1, -1, img_h, img_w)

        return coatt_hm_pred

    def calc_group_ap(self):
        coatt_level_gt = [output['coatt_level_gt'] for output in self.test_step_outputs]
        # coatt_hm_pred = [output['coatt_hm_pred'] for output in self.test_step_outputs]
        # coatt_hm_gt = [output['coatt_hm_gt'] for output in self.test_step_outputs]
        coatt_hm_gt_pk_cd = [output['coatt_hm_gt_pk_cd'] for output in self.test_step_outputs]
        coatt_hm_gt_pk_val = [output['coatt_hm_gt_pk_val'] for output in self.test_step_outputs]
        inout_pred_w_pad = [torch.sigmoid(output['inout_pred']) for output in self.test_step_outputs]
        hm_pred = [output['hm_pred'] for output in self.test_step_outputs]
        gaze_pts_pred = [output['gaze_pts_pred'] for output in self.test_step_outputs]

        # remove padding in inout prediction
        head_bboxes = [output['head_bboxes'] for output in self.test_step_outputs]
        inout_pred = []
        for idx in range(len(inout_pred_w_pad)):
            head_bboxes_sample = head_bboxes[idx]
            inout_pred_w_pad_sample = inout_pred_w_pad[idx]
            pad_mask = torch.sum(head_bboxes_sample, dim=-1) <= 0
            inout_pred_w_pad_sample[pad_mask] = 0.0
            inout_pred.append(inout_pred_w_pad_sample)
        # print(f"inout_pred: {inout_pred}")

        # for idx, output in enumerate(self.test_step_outputs):
        #     demo_coatt_level_pred = torch.sigmoid(output['coatt_level_pred'])[0, :, :]
        #     for i in range(demo_coatt_level_pred.shape[0]):
        #         coatt_level_pred_token_prob = (demo_coatt_level_pred[i, :])
        #         coatt_level_pred_token = coatt_level_pred_token_prob > 0.2
        #         if coatt_level_pred_token.sum()>=2:
        #             print(f"Co-attention levels for person {i}: {coatt_level_pred_token}")
            
        #     demo_coatt_level_gt = coatt_level_gt[idx][0, :, :]
        #     for i in range(demo_coatt_level_gt.shape[0]):
        #         coatt_level_gt_token = (demo_coatt_level_gt[i, :])
        #         if coatt_level_gt_token.sum()>=2:
        #             print(f"Ground-truth co-attention levels for person {i}: {coatt_level_gt_token}")

        # thresholds for group-level prediction
        coatt_level_thresh_list = [round(0.1 * i, 1) for i in range(1, 10, 2)]

        # thresholds for edge-based group detection
        edge_res_list = [round(0.1 * i, 1) for i in range(10, 15, 1)]

        # thresholds for post-processing
        hm_dist_thr_pp_list = [0.05, 0.1, 0.15, 0.2, 0.3, 100.0]

        # evaluation by varying different group IoU thresholds
        group_iou_thr_list = [0.5, 0.75, 1.0]
        hm_dist_thr_list = [0.05, 0.1, 0.2, 100.0]

        for group_iou_thr in group_iou_thr_list:
            for hm_dist_thr in hm_dist_thr_list:
                # group_ap = GroupAP(iou_thresh=group_iou_thr, hm_thresh=hm_dist_thr)
                group_ap = GroupAPFast(iou_thresh=group_iou_thr, hm_thresh=hm_dist_thr)

                # evaluation by using our group-level prediction
                # for co_lev_thr in coatt_level_thresh_list:
                    # print(f"Calculating group AP for group IoU threshold {group_iou_thresh}, co-attention level threshold {co_lev_thr}...")
                    # coatt_level_pred = [(torch.sigmoid(output['coatt_level_pred'])>co_lev_thr).int() for output in self.test_step_outputs]
                    # ret_metrics = group_ap.compute(coatt_level_pred, coatt_level_gt, coatt_hm_pred, coatt_hm_gt)
                    # self.log(f"metric/test/coatt_ap_grp_{group_iou_thr}_{hm_dist_thr}_{co_lev_thr}", ret_metrics['ap'], batch_size=1, prog_bar=True, on_step=False, on_epoch=True)
                    # self.log(f"metric/test/coatt_dist_grp_{group_iou_thr}_{hm_dist_thr}_{co_lev_thr}", ret_metrics['dist'], batch_size=1, prog_bar=True, on_step=False, on_epoch=True)
                
                # # evaluation by using edge-based group detection
                # for res in edge_res_list:
                #     coatt_level_pred_pairs = []
                #     coatt_hm_pred_pairs = []
                #     for idx in range(len(coatt_pred)):
                #         coatt_level_pred_pairs_sample = self.edge_base_grp_detection(coatt_pred[idx], coatt_level_gt[idx], inout_pred[idx], res)
                #         coatt_level_pred_pairs.append(coatt_level_pred_pairs_sample)
                #         coatt_hm_pred_pairs_sample = self.generate_coatt_hm_from_level(coatt_level_pred_pairs[idx], hm_pred[idx])
                #         coatt_hm_pred_pairs.append(coatt_hm_pred_pairs_sample)
                #     ret_metrics = group_ap.compute(coatt_level_pred_pairs, coatt_level_gt, coatt_hm_pred_pairs, coatt_hm_gt)
                #     self.log(f"metric/test/coatt_ap_pairs_{group_iou_thr}_{hm_dist_thr}_{res}", ret_metrics['ap'], batch_size=1, prog_bar=True, on_step=False, on_epoch=True)
                #     self.log(f"metric/test/coatt_dist_pairs_{group_iou_thr}_{hm_dist_thr}_{res}", ret_metrics['dist'], batch_size=1, prog_bar=True, on_step=False, on_epoch=True)

                # evaluation by using post-processing
                for dist_thr_pp in hm_dist_thr_pp_list:
                    coatt_level_pred_pp = self.post_process_coatt_pred(gaze_pts_pred, inout_pred, dist_thr_pp)
                    coatt_hm_pred_pp = []
                    coatt_hm_pred_pp_pk_cd = []
                    coatt_hm_pred_pp_pk_val = []

                    for idx in range(len(coatt_level_pred_pp)):
                        coatt_hm_pred_pp_sample = self.generate_coatt_hm_from_level(coatt_level_pred_pp[idx], hm_pred[idx])
                        coatt_hm_pred_pp.append(coatt_hm_pred_pp_sample)
                        coatt_hm_pred_pp_pk_cd_sample = spatial_argmax2d(coatt_hm_pred_pp_sample.reshape(-1, coatt_hm_pred_pp_sample.shape[2], coatt_hm_pred_pp_sample.shape[3]), normalize=True).view(1, -1, 2)
                        coatt_hm_pred_pp_pk_cd.append(coatt_hm_pred_pp_pk_cd_sample)
                        coatt_hm_pred_pp_pk_val_sample = torch.gather(coatt_hm_pred_pp_sample.reshape(-1, coatt_hm_pred_pp_sample.shape[2]*coatt_hm_pred_pp_sample.shape[3]), 1, (coatt_hm_pred_pp_pk_cd_sample[..., 0]*coatt_hm_pred_pp_sample.shape[3] + coatt_hm_pred_pp_pk_cd_sample[..., 1]).long())
                        coatt_hm_pred_pp_pk_val.append(coatt_hm_pred_pp_pk_val_sample)

                    # ret_metrics = group_ap.compute(coatt_level_pred_pp, coatt_level_gt, coatt_hm_pred_pp, coatt_hm_gt)
                    ret_metrics = group_ap.compute(coatt_level_pred_pp, coatt_level_gt, 
                                                    coatt_hm_pred_pp_pk_cd, coatt_hm_gt_pk_cd,
                                                    coatt_hm_pred_pp_pk_val, coatt_hm_gt_pk_val)

                    self.log(f"metric/test/coatt_ap_pp_{group_iou_thr}_{hm_dist_thr}_{dist_thr_pp}", ret_metrics['ap'], batch_size=1, prog_bar=True, on_step=False, on_epoch=True)
                    self.log(f"metric/test/coatt_dist_pp_{group_iou_thr}_{hm_dist_thr}_{dist_thr_pp}", ret_metrics['dist'], batch_size=1, prog_bar=True, on_step=False, on_epoch=True)

    def on_test_epoch_end(self):
        # Reset metrics
        self.metrics["test_dist"].reset()
        # self.metrics["test_auc"].reset()

        save_dir = os.path.join(os.path.dirname(self.cfg.test.checkpoint), self.cfg.experiment.dataset)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

        # Compute mAP
        print(f'Calculating group AP metrics...')
        self.calc_group_ap()

        # detach tensors from GPU
        for output in self.test_step_outputs:
            for key in output.keys():
                if isinstance(output[key], torch.Tensor):
                    output[key] = output[key].detach().cpu()

        # Save test predictions
        print(f'Saving test predictions...')
        self._save_predictions(self.test_step_outputs, save_dir)

        # Save test metrics
        print(f'Saving test metrics...')
        self._save_metrics(save_dir)

    def _save_metrics(self, save_dir):
        test_results = {}
        for key, value in self.trainer.logged_metrics.items():
            if key.startswith("metric/test/"):
                clean_key = key.replace("metric/test/", "")
                test_results[clean_key] = value.item()

        output_path = os.path.join(save_dir, "test_results.json")
        print(colored(f"Saving final test results to: {output_path}", "green"))
        with open(output_path, 'w') as f:
            json.dump(test_results, f, indent=4)

    def _save_predictions(self, outputs, save_dir):
        output_path = os.path.join(save_dir, "test-predictions.pickle")
        with open(output_path, "wb") as file:
            pickle.dump(outputs, file)