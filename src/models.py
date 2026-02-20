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
from tqdm import tqdm

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

from src.losses import compute_chong_loss, compute_sharingan_loss, compute_rinnegan_loss, compute_social_loss, compute_interact_loss, compute_inout_loss, compute_coatt_loss
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
from src.utils import generate_binary_gaze_heatmap, generate_gaze_heatmap, spatial_argmax2d
TERM_COLOR = "cyan"


# ==================================================================================================================
#                                                 INTERACT MODEL                                                  #
# ==================================================================================================================
class InteractModel(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        
        print('---------------------------------------------------------')
        print('Experiment: ', cfg.experiment.name)
        print('---------------------------------------------------------')

        self.model_name = cfg.model.model_name
        # Initialize model
        if self.model_name=='sharingan_social':
            self.model = Sharingan_social(    
                patch_size=cfg.model.sharingan.patch_size,
                token_dim=cfg.model.sharingan.token_dim,
                image_size=cfg.model.sharingan.image_size,
                gaze_feature_dim=cfg.model.sharingan.gaze_feature_dim,
                encoder_depth=cfg.model.sharingan.encoder_depth,
                encoder_num_heads=cfg.model.sharingan.encoder_num_heads,
                encoder_num_global_tokens=cfg.model.sharingan.encoder_num_global_tokens,
                encoder_mlp_ratio=cfg.model.sharingan.encoder_mlp_ratio,
                encoder_use_qkv_bias=cfg.model.sharingan.encoder_use_qkv_bias,
                encoder_drop_rate=cfg.model.sharingan.encoder_drop_rate,
                encoder_attn_drop_rate=cfg.model.sharingan.encoder_attn_drop_rate,
                encoder_drop_path_rate=cfg.model.sharingan.encoder_drop_path_rate,
                decoder_feature_dim=cfg.model.sharingan.decoder_feature_dim,
                decoder_hooks=cfg.model.sharingan.decoder_hooks,
                decoder_hidden_dims=cfg.model.sharingan.decoder_hidden_dims,
                decoder_use_bn=cfg.model.sharingan.decoder_use_bn,
                output=cfg.model.sharingan.output
            )
        elif self.model_name=='gaze_interact':
            self.model = InteractNet(  
                cfg=cfg,  
                patch_size=cfg.model.sharingan.patch_size,
                token_dim=cfg.model.sharingan.token_dim,
                image_size=cfg.model.sharingan.image_size,
                gaze_feature_dim=cfg.model.sharingan.gaze_feature_dim,
                encoder_depth=cfg.model.sharingan.encoder_depth,
                encoder_num_heads=cfg.model.sharingan.encoder_num_heads,
                encoder_num_global_tokens=cfg.model.sharingan.encoder_num_global_tokens,
                encoder_mlp_ratio=cfg.model.sharingan.encoder_mlp_ratio,
                encoder_use_qkv_bias=cfg.model.sharingan.encoder_use_qkv_bias,
                encoder_drop_rate=cfg.model.sharingan.encoder_drop_rate,
                encoder_attn_drop_rate=cfg.model.sharingan.encoder_attn_drop_rate,
                encoder_drop_path_rate=cfg.model.sharingan.encoder_drop_path_rate,
                decoder_feature_dim=cfg.model.sharingan.decoder_feature_dim,
                decoder_hooks=cfg.model.sharingan.decoder_hooks,
                decoder_hidden_dims=cfg.model.sharingan.decoder_hidden_dims,
                decoder_use_bn=cfg.model.sharingan.decoder_use_bn,
                temporal_context=cfg.data.temporal_context,
                output=cfg.model.sharingan.output,
                vlm_dim=cfg.data.vlm_dim,
                num_coatt=cfg.data.num_coatt,
            )
        elif self.model_name=='geom_gaze':
            self.model = GeomGaze(output_size=(cfg.data.heatmap_size[0],cfg.data.heatmap_size[1]))

        self.cfg = cfg
        self.output = cfg.model.sharingan.output
        self.num_tranining_samples = cfg.data.num_samples  # TODO: update this. VAT: ?, GazeFollow: 108955
        self.num_steps_in_epoch = math.ceil(self.num_tranining_samples / cfg.train.batch_size)
        self.test_step_outputs = []

        # Model weights Paths
        self.model_weights = cfg.model.weights
        self.gaze_weights = cfg.model.sharingan.gaze_weights
        self.multivit_weights = cfg.model.sharingan.multivit_weights

        # Define Metrics
        if cfg.experiment.dataset=='gazefollow':
            self.metrics = nn.ModuleDict({"val_dist": Distance(), "test_dist": GFTestDistance(), "test_auc": GFTestAUC()})
        else:
            self.metrics = nn.ModuleDict({"val_dist": Distance(), "test_dist": Distance(), "test_auc": AUC()})
        
        # Define Social Gaze Metrics
        self.val_coatt_auc = tm.AUROC(task="binary", ignore_index=-1)
        self.val_coatt_ap = tm.AveragePrecision(task="binary", ignore_index=-1)
        
        self.val_laeo_auc = tm.AUROC(task="binary", ignore_index=-1)
        self.val_laeo_ap = tm.AveragePrecision(task="binary", ignore_index=-1)
        
        self.val_lah_auc = tm.AUROC(task="binary", ignore_index=-1)
        self.val_lah_ap = tm.AveragePrecision(task="binary", ignore_index=-1)
        
        # Define Test Metrics
        self.test_coatt_auc = tm.AUROC(task="binary", ignore_index=-1)
        self.test_coatt_ap = tm.AveragePrecision(task="binary", ignore_index=-1)
        self.test_coatt_auc_grp = tm.AUROC(task="binary", ignore_index=-1)
        self.test_coatt_ap_grp = tm.AveragePrecision(task="binary", ignore_index=-1)

        self.test_laeo_auc = tm.AUROC(task="binary", ignore_index=-1)
        self.test_laeo_ap = tm.AveragePrecision(task="binary", ignore_index=-1)
        
        self.test_lah_auc = tm.AUROC(task="binary", ignore_index=-1)
        self.test_lah_ap = tm.AveragePrecision(task="binary", ignore_index=-1)

        # Define Loss Function
        self.compute_hm_loss = compute_interact_loss
        self.compute_dist_loss = compute_sharingan_loss
        self.compute_social_loss = compute_social_loss
        self.compute_speaking_loss = compute_inout_loss
        self.compute_coatt_loss = compute_coatt_loss

        # Initialize Weights
        self._init_weights()
        
        # Freeze Weights
        self._freeze()

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

    def _freeze(self):
        if self.cfg.train.freeze.gaze_encoder_backbone:
            print(colored(f"Freezing the Gaze Encoder backbone layers.", TERM_COLOR))
            self.freeze_module(self.model.gaze_encoder.backbone)
        if self.cfg.train.freeze.gaze_encoder:
            print(colored(f"Freezing the Gaze Encoder layers.", TERM_COLOR))
            self.freeze_module(self.model.gaze_encoder)
        if self.cfg.train.freeze.image_tokenizer:
            print(colored(f"Freezing the Image Tokenizer layers.", TERM_COLOR))
            self.freeze_module(self.model.image_tokenizer)
        if self.cfg.train.freeze.vit_encoder:
            print(colored(f"Freezing the ViT Encoder layers.", TERM_COLOR))
            self.freeze_module(self.model.encoder)
        if self.cfg.train.freeze.vit_adaptor:
            print(colored(f"Freezing the ViT Adaptor layers.", TERM_COLOR))
            self.freeze_module(self.model.vit_adaptor)

        # if self.cfg.train.freeze.gaze_decoder:
            # print(colored(f"Freezing the Gaze Decoder layers.", TERM_COLOR))
            # self.freeze_module(self.model.gaze_decoder)
        if self.cfg.train.freeze.gaze_decoder:
            print(colored(f"Freezing the Gaze Heatmap Decoder layers.", TERM_COLOR))
            self.freeze_module(self.model.gaze_hm_decoder_new)

        if self.cfg.train.freeze.inout_decoder:
            print(colored(f"Freezing the InOut Decoder layers.", TERM_COLOR))
            self.freeze_module(self.model.inout_decoder)

    def forward(self, batch):
        return self.model(batch)

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
            lr_scheduler = StepLR(optimizer, step_size=self.cfg.scheduler.t_0_epochs, gamma=0.1)
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
        if self.cfg.train.freeze.gaze_encoder:
            self.model.gaze_encoder.apply(self._set_batchnorm_eval)
        if self.cfg.train.freeze.image_tokenizer:
            self.model.image_tokenizer.apply(self._set_batchnorm_eval)
        if self.cfg.train.freeze.vit_encoder:
            self.model.encoder.apply(self._set_batchnorm_eval)

        # if self.cfg.train.freeze.gaze_decoder:
            # self.model.gaze_decoder.apply(self._set_batchnorm_eval)
        if self.cfg.train.freeze.gaze_decoder:
            self.model.gaze_hm_decoder_new.apply(self._set_batchnorm_eval)

        if self.cfg.train.freeze.inout_decoder:
            self.model.inout_decoder.apply(self._set_batchnorm_eval)
                
    def training_step(self, batch, batch_idx):
        nv = int((batch["speaking"]!=-1).sum().item())
        ni = int((batch["inout"]==1).sum().item())

        # Forward pass
        if self.output=='heatmap':
            out = self(batch)
            gaze_vec_pred = out['gaze_vec']
            gaze_hm_pred = out['gaze_hm']
            inout_pred = out['inout']
            lah_pred = out['lah']
            laeo_pred = out['laeo']
            coatt_pred = out['coatt']
            person_tokens = out['person_tokens']
            # coatt_hm_pred = out['coatt_hm']
            # coatt_level_pred = out['coatt_level']
            coatt_hm_all_pred = out['coatt_hm_all']
            coatt_level_all_pred = out['coatt_level_all']
            batch_size, t, n, hm_h, hm_w = gaze_hm_pred.shape
            gaze_hm_pred = gaze_hm_pred.view(batch_size*t, n, hm_h, hm_w)
        else:
            gaze_vec_pred, gaze_pt_pred, inout_pred, lah_pred, laeo_pred, coatt_pred = self(batch)
            batch_size, t, n = gaze_pt_pred.shape[:-1]
            gaze_pt_pred = gaze_pt_pred.view(batch_size*t, n, -1)
        gaze_vec_pred = gaze_vec_pred.view(batch_size*t, n, -1)
        inout_pred = inout_pred.view(batch_size*t, -1)
        lah_pred = lah_pred.view(batch_size*t, -1)
        laeo_pred = laeo_pred.view(batch_size*t, -1)
        coatt_pred = coatt_pred.view(batch_size*t, -1)

        # Compute distance, inout loss
        if self.output=='heatmap':
            if self.model_name=='geom_gaze':
                loss_dist, logs_dist = self.compute_hm_loss(batch["gaze_vecs_3d"].view(batch_size*t, n, -1), batch["gaze_heatmaps"].view(batch_size*t, n, hm_h, hm_w), batch["inout"].view(batch_size*t, -1), gaze_vec_pred, gaze_hm_pred, inout_pred)    # 3d gaze angle loss
            else:
                loss_dist, logs_dist = self.compute_hm_loss(batch["gaze_vecs"].view(batch_size*t, n, -1), batch["gaze_heatmaps"].view(batch_size*t, n, hm_h, hm_w), batch["inout"].view(batch_size*t, -1), gaze_vec_pred, gaze_hm_pred, inout_pred)    # 2d gaze angle loss
        else:
            loss_dist, logs_dist = self.compute_dist_loss(batch["gaze_vecs"], batch["gaze_pts"], batch["inout"].view(batch_size*t, -1), gaze_vec_pred, gaze_pt_pred, inout_pred)
        loss = loss_dist

        if self.cfg.train.social_loss:
            # Compute social gaze loss
            # coatt_gt, coatt_mask = id_to_pairwise_coatt(batch["coatt_ids"].view(batch_size*t, -1))
            # print('social-coatt loss time: ', time.time()-time_s)
            # coatt_gt_vec, coatt_mask_vec = id_to_pairwise_coatt_vectorized(batch["coatt_ids"].view(batch_size*t, -1))
            # print('social-coatt-vec loss time: ', time.time()-time_s)
            # check the two coatt implementations give the same result
            # if not torch.equal(coatt_gt, coatt_gt_vec):
                # print('Coatt GT not equal!')
            # if not torch.equal(coatt_mask, coatt_mask_vec):
                # print('Coatt Mask not equal!')
            coatt_gt, coatt_mask = id_to_pairwise_coatt(batch["coatt_ids"].view(batch_size*t, -1))

            # laeo_gt, laeo_mask = id_to_pairwise_laeo(batch["laeo_ids"].view(batch_size*t, -1))
            # print('social-laeo loss time: ', time.time()-time_s)
            # laeo_gt_vec, laeo_mask_vec = id_to_pairwise_laeo_vectorized(batch["laeo_ids"].view(batch_size*t, -1))
            # print('social-laeo-vec loss time: ', time.time()-time_s)
            # if not torch.equal(laeo_gt, laeo_gt_vec):
                # print('LAEO GT not equal!')
            # if not torch.equal(laeo_mask, laeo_mask_vec):
                # print('LAEO Mask not equal!')
            laeo_gt, laeo_mask = id_to_pairwise_laeo(batch["laeo_ids"].view(batch_size*t, -1))

            # lah_gt, lah_mask = id_to_pairwise_lah(batch["lah_ids"].view(batch_size*t, -1))
            # print('social-lah loss time: ', time.time()-time_s)
            # lah_gt_vec, lah_mask_vec = id_to_pairwise_lah_vectorized(batch["lah_ids"].view(batch_size*t, -1))
            # print('social-lah-vec loss time: ', time.time()-time_s)
            # if not torch.equal(lah_gt, lah_gt_vec):
                # print('LAH GT not equal!')
            # if not torch.equal(lah_mask, lah_mask_vec):
                # print('LAH Mask not equal!')
            lah_gt, lah_mask = id_to_pairwise_lah(batch["lah_ids"].view(batch_size*t, -1))

            loss_social, logs_social = self.compute_social_loss(lah_pred, lah_gt, lah_mask, laeo_pred, laeo_gt, laeo_mask, coatt_pred, coatt_gt, coatt_mask)
            loss += loss_social  
        
            # Log Social Gaze Losses
            self.log("loss/train/lah", logs_social["lah_loss"], batch_size=lah_mask.sum(), prog_bar=True, on_step=True, on_epoch=True)
            self.log("loss/train/laeo", logs_social["laeo_loss"], batch_size=laeo_mask.sum(), prog_bar=True, on_step=True, on_epoch=True)
            self.log("loss/train/coatt", logs_social["coatt_loss"], batch_size=coatt_mask.sum(), prog_bar=True, on_step=True, on_epoch=True)
 
        if self.cfg.train.coatt_loss:
            # Compute coatt loss
            coatt_ids = batch['coatt_ids']
            coatt_hm_gt = batch['coatt_heatmaps']
            coatt_level_gt = batch['coatt_levels']
            # coatt_loss, logs_coatt = self.compute_coatt_loss(coatt_hm_gt, coatt_hm_pred, coatt_level_gt, coatt_level_pred, person_tokens)
            coatt_loss, con_loss, logs_coatt = self.compute_coatt_loss(coatt_hm_gt, coatt_hm_all_pred, coatt_level_gt, coatt_level_all_pred,
                                                                        person_tokens, self.cfg)
            loss += coatt_loss

            if self.cfg.train.coatt_con_loss:
                loss += con_loss

            # Log Coatt Losses
            self.log("loss/train/coatt_hm", logs_coatt["coatt_hm_loss"], batch_size=ni, prog_bar=False, on_step=True, on_epoch=True)
            self.log("loss/train/coatt_level", logs_coatt["coatt_level_loss"], batch_size=ni, prog_bar=False, on_step=True, on_epoch=True)
            self.log("loss/train/coatt_con", logs_coatt["con_loss"], batch_size=ni, prog_bar=False, on_step=True, on_epoch=True)
        
        # Logging Distance, InOut losses            
        self.log("loss/train/heatmap", logs_dist["heatmap_loss"], batch_size=ni, prog_bar=False, on_step=True, on_epoch=True)
        # self.log("loss/train/dist", logs_dist["dist_loss"], batch_size=ni, prog_bar=False, on_step=True, on_epoch=True)
        self.log("loss/train/angular", logs_dist["angular_loss"], batch_size=ni, prog_bar=False, on_step=True, on_epoch=True)
        self.log("loss/train/inout", logs_dist["inout_loss"], batch_size=n, prog_bar=False, on_step=True, on_epoch=True)
        self.log("loss/train", loss.item(), batch_size=n, prog_bar=True, on_step=True, on_epoch=True)

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        # nv = int((batch["speaking"]!=-1).sum().item())
        ni = int((batch["inout"]==1).sum().item())

        # Forward pass
        if self.output=='heatmap':
            out = self(batch)
            gaze_vec_pred = out['gaze_vec']
            gaze_hm_pred = out['gaze_hm']
            inout_pred = out['inout']
            lah_pred = out['lah']
            laeo_pred = out['laeo']
            coatt_pred = out['coatt']
            person_tokens = out['person_tokens']
            # coatt_hm_pred = out['coatt_hm']
            # coatt_level_pred = out['coatt_level']
            coatt_hm_all_pred = out['coatt_hm_all']
            coatt_level_all_pred = out['coatt_level_all']

            # only take outputs of central frame
            batch_size, t, n, hm_h, hm_w = gaze_hm_pred.shape
            middle_frame_idx = int(t/2)
            gaze_hm_pred = gaze_hm_pred[:,middle_frame_idx,:,:,:]
            # perform argmax for gaze point
            gaze_pt_pred = spatial_argmax2d(gaze_hm_pred.reshape(batch_size*n, hm_h, hm_w), normalize=True).view(batch_size, n, -1)
        else:
            gaze_vec_pred, gaze_pt_pred, inout_pred, lah_pred, laeo_pred, coatt_pred = self(batch)
            batch_size, t, n = gaze_pt_pred.shape[:-1]
            middle_frame_idx = int(t/2)
            gaze_pt_pred = gaze_pt_pred[:,middle_frame_idx,:,:]

        gaze_vec_pred = gaze_vec_pred[:,middle_frame_idx,:,:]
        inout_pred = inout_pred[:,middle_frame_idx,:]
        lah_pred = lah_pred[:,middle_frame_idx,:]
        laeo_pred = laeo_pred[:,middle_frame_idx,:]
        coatt_pred = coatt_pred[:,middle_frame_idx,:]
        # coatt_hm_pred = coatt_hm_pred[:,middle_frame_idx,:,:,:]
        # coatt_level_pred = coatt_level_pred[:,middle_frame_idx,:,:]
        coatt_hm_all_pred = coatt_hm_all_pred[:,:, middle_frame_idx,:,:]
        coatt_level_all_pred = coatt_level_all_pred[:,:, middle_frame_idx,:,:]

        # Compute dist, inout loss
        if self.output=='heatmap':
            if self.model_name=='geom_gaze':
                loss_dist, logs_dist = self.compute_hm_loss(batch["gaze_vecs_3d"][:,middle_frame_idx,:, :], batch["gaze_heatmaps"][:,middle_frame_idx,:,:,:], batch["inout"][:,middle_frame_idx,:], gaze_vec_pred, gaze_hm_pred, inout_pred)    # 3d gaze vector loss
            else:
                loss_dist, logs_dist = self.compute_hm_loss(batch["gaze_vecs"][:,middle_frame_idx,:, :], batch["gaze_heatmaps"][:,middle_frame_idx,:,:,:], batch["inout"][:,middle_frame_idx,:], gaze_vec_pred, gaze_hm_pred, inout_pred)    # 2d gaze vector loss
        else:
            loss_dist, logs_dist = self.compute_dist_loss(batch["gaze_vecs"], batch["gaze_pts"], batch["inout"][:,middle_frame_idx,:], gaze_vec_pred, gaze_pt_pred, inout_pred)
        
        loss = loss_dist
        # Logging losses
        self.log("loss/val/heatmap", logs_dist["heatmap_loss"], batch_size=ni, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        # self.log("loss/val/dist", logs_dist["dist_loss"], batch_size=ni, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        self.log("loss/val/angular", logs_dist["angular_loss"], batch_size=ni, prog_bar=False, on_step=True, on_epoch=True, sync_dist=True)
        self.log("loss/val/inout", logs_dist["inout_loss"], batch_size=n, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        self.log("loss/val", loss.item(), batch_size=n, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        
        # Update dist metrics
        # self.metrics["val_auc"].update(gaze_heatmap_pred, gaze_heatmap, inout)
        self.metrics["val_dist"].update(gaze_pt_pred, batch["gaze_pts"][:,middle_frame_idx,:, :], batch["inout"][:,middle_frame_idx,:])
        # self.log("metric/val/auc", self.metrics["val_auc"], batch_size=ni, prog_bar=True, on_step=False, on_epoch=True)
        self.log("metric/val/dist", self.metrics["val_dist"], batch_size=ni, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        
        if self.cfg.train.social_loss:
            # Compute social gaze loss
            coatt_gt, coatt_mask = id_to_pairwise_coatt(batch["coatt_ids"][:,middle_frame_idx,:])
            laeo_gt, laeo_mask = id_to_pairwise_laeo(batch["laeo_ids"][:,middle_frame_idx,:])
            lah_gt, lah_mask = id_to_pairwise_lah(batch["lah_ids"][:,middle_frame_idx,:])
            loss_social, logs_social = self.compute_social_loss(lah_pred, lah_gt, lah_mask, laeo_pred, laeo_gt, laeo_mask, coatt_pred, coatt_gt, coatt_mask)
            loss += loss_social        
            coatt_gt = torch.where(coatt_mask, coatt_gt, torch.tensor(-1., device=coatt_mask.device)).int()
            laeo_gt = torch.where(laeo_mask, laeo_gt, torch.tensor(-1., device=laeo_mask.device)).int()
            lah_gt = torch.where(lah_mask, lah_gt, torch.tensor(-1., device=lah_mask.device)).int()
        
            # Log Social Gaze Losses
            self.log("loss/val/lah", logs_social["lah_loss"], batch_size=lah_mask.sum(), prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
            self.log("loss/val/laeo", logs_social["laeo_loss"], batch_size=laeo_mask.sum(), prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
            self.log("loss/val/coatt", logs_social["coatt_loss"], batch_size=coatt_mask.sum(), prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)

            # Update LAEO metrics
            if laeo_pred.sum()!=0:
                laeo_pred = torch.sigmoid(laeo_pred)
                if laeo_mask.sum()>0:
                    self.val_laeo_auc(laeo_pred, laeo_gt)
                    self.val_laeo_ap(laeo_pred, laeo_gt)

                    self.log("metric/val/laeo_auc", self.val_laeo_auc, batch_size=laeo_mask.sum(), prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
                    self.log("metric/val/laeo_ap", self.val_laeo_ap, batch_size=laeo_mask.sum(), prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
            
            # Update LAH metrics
            if lah_pred.sum()!=0:
                lah_pred = torch.sigmoid(lah_pred)
                if lah_mask.sum()>0:
                    self.val_lah_auc(lah_pred, lah_gt)
                    self.val_lah_ap(lah_pred, lah_gt)

                    self.log("metric/val/lah_auc", self.val_lah_auc, batch_size=lah_mask.sum(), prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
                    self.log("metric/val/lah_ap", self.val_lah_ap, batch_size=lah_mask.sum(), prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)

        if self.cfg.train.coatt_loss:
            # Compute coatt loss
            coatt_hm_gt = batch['coatt_heatmaps'][:,middle_frame_idx,:,:,:]
            coatt_levels_gt = batch['coatt_levels'][:,middle_frame_idx,:,:]
            # coatt_loss, con_loss, logs_coatt = self.compute_coatt_loss(coatt_hm_gt, coatt_hm_pred, coatt_levels_gt, coatt_level_pred, person_tokens, self.cfg)
            coatt_loss, con_loss, logs_coatt = self.compute_coatt_loss(coatt_hm_gt, coatt_hm_all_pred, coatt_levels_gt, coatt_level_all_pred, person_tokens, self.cfg)
            loss += coatt_loss

            if self.cfg.train.coatt_con_loss:
                loss += con_loss

            # Log Coatt Losses
            self.log("loss/val/coatt_hm", logs_coatt["coatt_hm_loss"], batch_size=ni, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
            self.log("loss/val/coatt_level", logs_coatt["coatt_level_loss"], batch_size=ni, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
            self.log("loss/val/coatt_con", logs_coatt["con_loss"], batch_size=ni, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
    
        # Update CoAtt Metrics
        if self.cfg.train.social_loss:
            if coatt_pred.sum()!=0:
                coatt_pred = torch.sigmoid(coatt_pred)
                if coatt_mask.sum()>0:
                    self.val_coatt_auc(coatt_pred, coatt_gt)
                    self.val_coatt_ap(coatt_pred, coatt_gt)

                    self.log("metric/val/coatt_auc", self.val_coatt_auc, batch_size=coatt_mask.sum(), prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
                    self.log("metric/val/coatt_ap", self.val_coatt_ap, batch_size=coatt_mask.sum(), prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
            

    def test_step(self, batch, batch_idx):
        ni = int((batch["inout"]==1).sum().item())
        # assert n == ni, f"Expected all test samples to be looking inside. Got {n} samples, {ni} of which are looking inside."

        # Forward pass
        if self.output=='heatmap':
            out = self(batch)
            gaze_vec_pred = out['gaze_vec']
            gaze_hm_pred = out['gaze_hm']
            inout_pred = out['inout']
            lah_pred = out['lah']
            laeo_pred = out['laeo']
            coatt_pred = out['coatt']
            # coatt_hm_pred = out['coatt_hm']
            # coatt_level_pred = out['coatt_level']
            coatt_hm_all_pred = out['coatt_hm_all']
            coatt_level_all_pred = out['coatt_level_all']

            batch_size, t, num_people, hm_h, hm_w = gaze_hm_pred.shape
            # only take outputs of central frame
            middle_frame_idx = int(t/2)
            gaze_hm_pred = gaze_hm_pred[:,middle_frame_idx,:,:,:]
            # perform arg max for gaze point
            gaze_pt_pred = spatial_argmax2d(gaze_hm_pred.reshape(batch_size*num_people, hm_h, hm_w), normalize=True).view(batch_size, num_people, -1)
        else:
            gaze_vec_pred, gaze_pt_pred, inout_pred, lah_pred, laeo_pred, coatt_pred = self(batch)
            batch_size, t, num_people = gaze_pt_pred.shape[:-1]
            middle_frame_idx = int(t/2)
            gaze_pt_pred = gaze_pt_pred[:,middle_frame_idx,:,:]
        gaze_vec_pred = gaze_vec_pred[:,middle_frame_idx,:,:]
        inout_pred = inout_pred[:,middle_frame_idx,:]
        lah_pred = lah_pred[:,middle_frame_idx,:]
        laeo_pred = laeo_pred[:,middle_frame_idx,:]
        coatt_pred = coatt_pred[:,middle_frame_idx,:]
        # coatt_hm_pred = coatt_hm_pred[:,middle_frame_idx,:,:,:]
        # coatt_level_pred = coatt_level_pred[:,middle_frame_idx,:,:]
        coatt_hm_all_pred = coatt_hm_all_pred[:,:, middle_frame_idx,:,:]
        coatt_level_all_pred = coatt_level_all_pred[:,:, middle_frame_idx,:,:]
        coatt_hm_pred = coatt_hm_all_pred[-1,:,:,:]
        coatt_level_pred = coatt_level_all_pred[-1,:,:]

        # Update distance metrics
        if self.cfg.experiment.dataset=='gazefollow':
            gaze_vec_pred = gaze_vec_pred[:, -1, :]  # (b, n, 2) >> (b, 2)
            gaze_pt_pred = gaze_pt_pred[:, -1, :]  # (b, n, 2) >> (b, 2)
            inout_pred = inout_pred[:, -1]  # (b, n) >> (b,)
            inout_gt = batch["inout"][:,middle_frame_idx]
            gaze_hm_pred = gaze_hm_pred[:, -1, :, :]

            test_auc = self.metrics["test_auc"](gaze_hm_pred, batch["gaze_pts"][:,middle_frame_idx,:, :])
            test_dist_to_avg, test_avg_dist, test_min_dist = self.metrics["test_dist"](gaze_pt_pred, batch["gaze_pts"][:,middle_frame_idx,:, :])
            # Log metrics
            # self.log("metric/test/auc", test_auc, batch_size=ni, prog_bar=True, on_step=False, on_epoch=True)
            # self.log("metric/test/dist_to_avg", test_dist_to_avg, batch_size=ni, prog_bar=True, on_step=False, on_epoch=True)
            # self.log("metric/test/avg_dist", test_avg_dist, batch_size=ni, prog_bar=True, on_step=False, on_epoch=True)
            # self.log("metric/test/min_dist", test_min_dist, batch_size=ni, prog_bar=True, on_step=False, on_epoch=True)
        else:
            inout_gt = batch["inout"][:,middle_frame_idx,:]   
            # Log metrics
            if self.output=='heatmap':
                test_auc = self.metrics["test_auc"](gaze_hm_pred.reshape(batch_size*num_people, hm_h, hm_w), batch["gaze_heatmaps"][:,middle_frame_idx,:,:,:].reshape(batch_size*num_people, hm_h, hm_w), inout_gt.reshape(batch_size*num_people, -1))
                # self.log("metric/test/auc", test_auc, batch_size=ni, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
            
            # self.metrics["test_dist"].update(gaze_pt_pred, batch["gaze_pts"][:,middle_frame_idx,:, :], inout_gt)
            
            gaze_pt_pred_mask = gaze_pt_pred
            gaze_pt_gt_mask = batch["gaze_pts"][:,middle_frame_idx,:, :]
            
            pt_gt_mask = torch.sum(gaze_pt_gt_mask==-1, dim=-1) != 2
            gaze_pt_pred_mask = gaze_pt_pred_mask[pt_gt_mask]
            gaze_pt_gt_mask = gaze_pt_gt_mask[pt_gt_mask]

            inout_gt = inout_gt == 1
            mask = inout_gt | pt_gt_mask
            mask = torch.where(mask==True, 1, -1)

            self.metrics["test_dist"].update(gaze_pt_pred, batch["gaze_pts"][:,middle_frame_idx,:, :], mask)
            self.log("metric/test/dist", self.metrics["test_dist"], batch_size=ni, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        
        coatt_gt, coatt_mask = id_to_pairwise_coatt(batch["coatt_ids"][:,middle_frame_idx,:])
        coatt_gt = torch.where(coatt_mask, coatt_gt, torch.tensor(-1., device=coatt_mask.device)).int()
        laeo_gt, laeo_mask = id_to_pairwise_laeo(batch["laeo_ids"][:,middle_frame_idx,:])
        laeo_gt = torch.where(laeo_mask, laeo_gt, torch.tensor(-1., device=laeo_mask.device)).int()
        lah_gt, lah_mask = id_to_pairwise_lah(batch["lah_ids"][:,middle_frame_idx,:])
        lah_gt = torch.where(lah_mask, lah_gt, torch.tensor(-1., device=lah_mask.device)).int()

        # Update CoAtt Metrics
        if coatt_pred.sum()!=0:
            coatt_pred = torch.sigmoid(coatt_pred)
            if coatt_mask.sum()>0:
                # auc and ap based on pair-wise estimation
                self.test_coatt_auc(coatt_pred, coatt_gt)
                self.test_coatt_ap(coatt_pred, coatt_gt)
                # self.log("metric/test/coatt_auc", self.test_coatt_auc, batch_size=coatt_mask.sum(), prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
                # self.log("metric/test/coatt_ap", self.test_coatt_ap, batch_size=coatt_mask.sum(), prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)

                # auc and ap based on group-wise estimation
                coatt_level_pred_prob = torch.sigmoid(coatt_level_pred)
                _, num_people = batch["coatt_ids"][:,middle_frame_idx,:].shape
                indices = torch.tensor(list(itertools.permutations(torch.arange(num_people), 2)))
                coatt_pred_grp = torch.zeros_like(coatt_pred)
                for ind in range(indices.shape[0]):
                    i, j = indices[ind]
                    coatt_pred_grp[:, ind] = torch.sigmoid(coatt_level_pred[:, :, [i, j]]).mean(dim=-1).max(dim=-1).values

                self.test_coatt_auc_grp(coatt_pred_grp, coatt_gt)
                self.test_coatt_ap_grp(coatt_pred_grp, coatt_gt)
                # self.log("metric/test/coatt_auc_grp", self.test_coatt_auc_grp, batch_size=coatt_mask.sum(), prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
                # self.log("metric/test/coatt_ap_grp", self.test_coatt_ap_grp, batch_size=coatt_mask.sum(), prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)

        if 'coatt_heatmaps' in batch and 'coatt_levels' in batch:
            coatt_levels_gt = batch['coatt_levels'][:,middle_frame_idx,:,:]
            coatt_hm_gt = batch['coatt_heatmaps'][:,middle_frame_idx,:,:,:]

            # >>>>>>>>> grouping metrics based on pair-wise estimation <<<<<<<<<<
            metrics_pairwise = {}
            coatt_level_pred_label_pairs_all = {}
            coatt_hm_pred_pairs_all = {}
            _, num_people = batch["coatt_ids"][:,middle_frame_idx,:].shape
            indices = torch.tensor(list(itertools.permutations(torch.arange(num_people), 2)))
            batch_size, token_size = coatt_levels_gt.shape[0], coatt_levels_gt.shape[1]
            
            # fill zero if the head is padded
            head_bboxes = batch['head_bboxes'][:,middle_frame_idx,:,:]
            head_pad_mask = torch.sum(head_bboxes, dim=-1) <= 0
            for ind in range(indices.shape[0]):
                i, j = indices[ind]
                coatt_pred[:, ind] = torch.where(head_pad_mask[:, i] | head_pad_mask[:, j], 0.001, coatt_pred[:, ind])

            # build graph based on pair-wise co-attention prediction
            edges = []
            for ind in range(indices.shape[0]):
                i, j = indices[ind]
                edges.append((i.item(), j.item(), {'weight': coatt_pred[0, ind].item()}))
            G = nx.Graph()
            G.add_edges_from(edges)

            resolution_list = [round(0.1 * i, 1) for i in range(1, 15, 2)]
            for res in resolution_list:
                partition = community.best_partition(G, weight='weight', resolution=res)
                coatt_level_pred_label_pairs = torch.zeros_like(coatt_levels_gt)
                group_ids = list(set(partition.values()))
                node_ids = list(partition.keys())
                for g_id in group_ids:
                    members = [node for node, comm_id in partition.items() if comm_id == g_id]
                    if len(members)>1 and len(members)<=token_size:
                        coatt_level_pred_label_pairs[0, g_id, members] = 1

                group_cost = GroupCost()
                coatt_hm_pred_pairs_sample = self.generate_coatt_hm_from_level(coatt_level_pred_label_pairs, gaze_hm_pred)
                ret_metrics_pairwse = group_cost.compute(coatt_level_pred_label_pairs, coatt_levels_gt, coatt_hm_pred_pairs_sample, coatt_hm_gt)
                metrics_pairwise[res] = [ret_metrics_pairwse['Group_IoU'], ret_metrics_pairwse['Group_Dist']]
                self.log(f"metric/test/coatt_cost_iou_pairs_{res}", ret_metrics_pairwse['Group_IoU'], batch_size=1, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
                self.log(f"metric/test/coatt_cost_dist_pairs_{res}", ret_metrics_pairwse['Group_Dist'], batch_size=1, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
                coatt_level_pred_label_pairs_all[res] = coatt_level_pred_label_pairs
                coatt_hm_pred_pairs_all[res] = coatt_hm_pred_pairs_sample

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
                    if pi in grp_pis or out_mask[pi]:
                        continue
                    gaze_pt_pred_p = gaze_pt_pred[0, pi]
                    dists_xy = gaze_pt_pred[0] - gaze_pt_pred_p
                    dists = torch.norm(dists_xy, dim=-1)
                    dists[out_mask] = float('inf')
                    close_indices = torch.where((dists<dist_thr).int())[0]
                    if len(close_indices)>1:
                        coatt_level_pred_grp = torch.zeros((num_people), device=gaze_pt_pred.device)
                        coatt_level_pred_grp[close_indices] = 1
                        coatt_level_pred_label_pp.append(coatt_level_pred_grp)
                        # grp_pis = grp_pis | set(close_indices.tolist())
                        grp_pis = grp_pis.union(set(close_indices.tolist()))
                
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

            # >>>>>>>>> grouping metrics based on group-wise estimation <<<<<<<<<<
            metrics_grp = {}
            coatt_level_thresh_list = [round(0.1 * i, 1) for i in range(1, 10, 2)]
            coatt_level_thresh_list += [0.05]
            for co_lev_thr in coatt_level_thresh_list:
                coatt_level_pred_label = (torch.sigmoid(coatt_level_pred)>co_lev_thr).int()
                # fill zero if inout is low
                # inout_low_indices = torch.where(inout_pred[0]<inout_thr)[0]
                # coatt_level_pred_label[:, inout_low_indices] = 0
                group_cost = GroupCost()
                ret_metrics_grp = group_cost.compute(coatt_level_pred_label, coatt_levels_gt, coatt_hm_pred, coatt_hm_gt)
                metrics_grp[co_lev_thr] = [ret_metrics_grp['Group_IoU'], ret_metrics_grp['Group_Dist']]
                self.log(f"metric/test/coatt_cost_iou_grp_{co_lev_thr}", ret_metrics_grp['Group_IoU'], batch_size=1, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
                self.log(f"metric/test/coatt_cost_dist_grp_{co_lev_thr}", ret_metrics_grp['Group_Dist'], batch_size=1, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)

        pair_indices = torch.tensor(list(itertools.permutations(torch.arange(num_people), 2)))
        # Update LAEO metrics
        if laeo_pred.sum()!=0:
            laeo_pred = torch.sigmoid(laeo_pred)
            # peform arg max for laeo
            laeo_pred_argmax = torch.zeros_like(laeo_pred)
            for bi in range(batch_size):
                for pi in range(num_people):
                    valid_indices = torch.where((pair_indices[:, 1]==pi).int() * (pair_indices[:, 0]!=0).int())[0]
                    if valid_indices.shape[0]>0:
                        max_val, max_idx = torch.max(laeo_pred[bi][valid_indices], 0)
                        laeo_pred_argmax[bi][valid_indices[max_idx]] = max_val
            if laeo_mask.sum()>0:
                self.test_laeo_auc(laeo_pred_argmax, laeo_gt)
                self.test_laeo_ap(laeo_pred_argmax, laeo_gt)
                # self.log("metric/test/laeo_auc", self.test_laeo_auc, batch_size=laeo_mask.sum(), prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
                # self.log("metric/test/laeo_ap", self.test_laeo_ap, batch_size=laeo_mask.sum(), prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        
        # Update LAH metrics
        if lah_pred.sum()!=0:
            lah_pred = torch.sigmoid(lah_pred)
            # peform arg max for lah
            lah_pred_argmax = torch.zeros_like(lah_pred)
            lah_gt_metric = torch.zeros(batch_size, num_people).long() - 1
            lah_pred_metric = torch.zeros(batch_size, num_people)
            for bi in range(batch_size):
                for pi in range(num_people):
                    if self.cfg.experiment.dataset=='gazefollow':
                        io = 1
                    else:
                        io = batch['inout'][bi][middle_frame_idx][pi]==1
                    if io==1:
                        valid_indices = torch.where((pair_indices[:, 1]==pi).int())[0]
                        if valid_indices.shape[0]>0:
                            if (lah_gt[bi][valid_indices]!=-1).sum()==0:
                                continue

                            max_val, max_idx = torch.max(lah_pred[bi][valid_indices], 0)
                            lah_pred_argmax[bi][valid_indices[max_idx]] = max_val

                            lah_gt_metric[bi][pi] = lah_gt[bi][valid_indices][lah_gt[bi][valid_indices]!=-1].sum()
                            gt_idx = torch.where(lah_gt[bi][valid_indices]==1)[0]
                            if len(gt_idx)>0:
                                lah_pred_metric[bi][pi] = lah_pred_argmax[bi][valid_indices][gt_idx]
                            else:
                                lah_pred_metric[bi][pi] = max_val
            if lah_mask.sum()>0 and (batch['inout'][:, middle_frame_idx]==1).sum()>0:
                self.test_lah_auc(lah_pred_metric, lah_gt_metric)
                self.test_lah_ap(lah_pred_metric, lah_gt_metric)

                # self.log("metric/test/lah_auc", self.test_lah_auc, batch_size=(lah_gt_metric!=-1).sum(), prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
                # self.log("metric/test/lah_ap", self.test_lah_ap, batch_size=(lah_gt_metric!=-1).sum(), prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        
        # Obtain the best metrics in each method. The threshold having the lowest distance metric is chosen to report the IoU metric.
        # pair-wise
        best_res_pairwise = max(metrics_pairwise, key=lambda x: metrics_pairwise[x][0])
        metrics_pairwise_best = metrics_pairwise[best_res_pairwise]
        coatt_level_pred_label_pairs_best = coatt_level_pred_label_pairs_all[best_res_pairwise]
        coatt_hm_pred_pairs_best = coatt_hm_pred_pairs_all[best_res_pairwise]
        # post-processing
        best_res_pp = max(metrics_pp, key=lambda x: metrics_pp[x][0])
        metrics_pp_best = metrics_pp[best_res_pp]
        coatt_level_pred_label_pp_best = coatt_level_pred_grp_all_pp[best_res_pp]
        coatt_hm_pred_pp_best = coatt_hm_pred_grp_all_pp[best_res_pp]
        # group-wise
        best_res_grp = max(metrics_grp, key=lambda x: metrics_grp[x][0])
        metrics_grp_best = metrics_grp[best_res_grp]

        # obtain coatt_hm peak_coordinates
        coatt_hm_gt_pk_cd = spatial_argmax2d(coatt_hm_gt.reshape(-1, coatt_hm_gt.shape[2], coatt_hm_gt.shape[3]), normalize=True).view(batch_size, -1, 2)
        coatt_hm_pred_pk_cd = spatial_argmax2d(coatt_hm_pred.reshape(-1, coatt_hm_pred.shape[2], coatt_hm_pred.shape[3]), normalize=True).view(batch_size, -1, 2)
        coatt_hm_pred_pairs_pk_cd = spatial_argmax2d(coatt_hm_pred_pairs_best.reshape(-1, coatt_hm_pred_pairs_best.shape[2], coatt_hm_pred_pairs_best.shape[3]), normalize=True).view(batch_size, -1, 2)
        coatt_hm_pred_pp_pk_cd = spatial_argmax2d(coatt_hm_pred_pp_best.reshape(-1, coatt_hm_pred_pp_best.shape[2], coatt_hm_pred_pp_best.shape[3]), normalize=True).view(batch_size, -1, 2)

        # obtain coatt_hm peak values
        coatt_hm_gt_pk_val = coatt_hm_gt.reshape(-1, coatt_hm_gt.shape[2]*coatt_hm_gt.shape[3]).max(dim=-1).values.view(batch_size, -1)
        coatt_hm_pred_pk_val = coatt_hm_pred.reshape(-1, coatt_hm_pred.shape[2]*coatt_hm_pred.shape[3]).max(dim=-1).values.view(batch_size, -1)

        # ontain coatt hm peak coordinates and values for coatt_lvel_all_pred
        coatt_hm_pred_pk_cd_all = []
        coatt_hm_pred_pk_val_all = []
        for i in range(coatt_hm_all_pred.shape[0]):
            coatt_hm_pred_pk_cd_i = spatial_argmax2d(coatt_hm_all_pred[i].reshape(-1, coatt_hm_all_pred.shape[3], coatt_hm_all_pred.shape[4]), normalize=True).view(batch_size, -1, 2)
            coatt_hm_pred_pk_val_i = coatt_hm_all_pred[i].reshape(-1, coatt_hm_all_pred.shape[3]*coatt_hm_all_pred.shape[4]).max(dim=-1).values.view(batch_size, -1)
            coatt_hm_pred_pk_cd_all.append(coatt_hm_pred_pk_cd_i)
            coatt_hm_pred_pk_val_all.append(coatt_hm_pred_pk_val_i)
        coatt_hm_pred_pk_cd_all = torch.stack(coatt_hm_pred_pk_cd_all, dim=0)
        coatt_hm_pred_pk_val_all = torch.stack(coatt_hm_pred_pk_val_all, dim=0)

        # Build output dict
        output = {
                  "head_bboxes": batch["head_bboxes"][:,middle_frame_idx,:, :],
                  "gp_pred": gaze_pt_pred, 
                  "gp_gt": batch["gaze_pts"][:,middle_frame_idx,:, :],
                  "gv_pred": gaze_vec_pred, 
                  "gv_gt": batch["gaze_vecs"][:,middle_frame_idx,:, :],
                  "gaze_pts_pred" : gaze_pt_pred,
                  "gaze_pts_gt" : batch["gaze_pts"][:,middle_frame_idx,:, :],
                #   # optionally save gaze heatmaps
                  "hm_pred": gaze_hm_pred,
                #   "hm_gt": batch["gaze_heatmaps"][:,middle_frame_idx,:,:,:],
                  "inout_gt": inout_gt, 
                  "path": batch["path"],
                  "inout_pred": inout_pred,
                  "coatt_pred": coatt_pred,
                #   "coatt_hm_pred": coatt_hm_pred,
                #   "coatt_hm_gt": coatt_hm_gt,
                  "coatt_hm_pred_pk_cd": coatt_hm_pred_pk_cd,
                  "coatt_hm_pred_pairs_pk_cd": coatt_hm_pred_pairs_pk_cd,
                  "coatt_hm_pred_pp_pk_cd": coatt_hm_pred_pp_pk_cd,
                  "coatt_hm_pred_pk_val": coatt_hm_pred_pk_val,
                  "coatt_hm_gt_pk_cd": coatt_hm_gt_pk_cd,
                  "coatt_hm_gt_pk_val": coatt_hm_gt_pk_val,
                  "coatt_level_pred": coatt_level_pred,
                  "coatt_level_pred_all": coatt_level_all_pred,
                  "coatt_hm_pred_pk_cd_all": coatt_hm_pred_pk_cd_all,
                  "coatt_hm_pred_pk_val_all": coatt_hm_pred_pk_val_all,
                  "coatt_level_pred_label_pairs_best": coatt_level_pred_label_pairs_best,
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
                  "group_iou_pairwise": metrics_pairwise_best[0],
                  "group_dist_pairwise": metrics_pairwise_best[1],
                  "best_res_pairwise": best_res_pairwise,
                  "group_iou_pp": metrics_pp_best[0],
                  "group_dist_pp": metrics_pp_best[1],
                  "best_dist_pp": best_res_pp,
                  "group_iou_grp": metrics_grp_best[0],
                  "group_dist_grp": metrics_grp_best[1],
                  "best_coatt_level_thresh_grp": best_res_grp,
                  "path": batch["path"],
                  }
        self.test_step_outputs.append(output)

    def edge_base_grp_detection_original(self, coatt_pred: torch.Tensor, coatt_level_gt: torch.Tensor, inout_pred: torch.Tensor, res: float) -> torch.Tensor:
        """Original (non-optimized) version for edge-based group detection"""
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

    def edge_base_grp_detection(self, coatt_pred: torch.Tensor, coatt_level_gt: torch.Tensor, inout_pred: torch.Tensor, res: float) -> torch.Tensor:
        """Optimized version for edge-based group detection using vectorized edge construction"""
        num_people = coatt_level_gt.shape[-1]
        
        # Construct edge list without explicit loops
        # Generate all pairs (i, j) where i < j
        edge_indices = torch.triu_indices(num_people, num_people, offset=1)
        i_indices = edge_indices[0]
        j_indices = edge_indices[1]
        
        # Get weights corresponding to these pairs
        # Map (i, j) pairs to the coatt_pred indices
        # For permutations(range(n), 2), order is (0,1), (0,2), ..., (0,n-1), (1,0), (1,2), ...
        # We need to find the mapping from our pair indices to coatt_pred
        perm_indices = torch.tensor(list(itertools.permutations(torch.arange(num_people), 2)))
        pair_to_perm = {}
        for perm_idx, (pi, pj) in enumerate(perm_indices):
            pair_to_perm[(pi.item(), pj.item())] = perm_idx
        
        # Create edges more efficiently
        edges = []
        for idx in range(len(i_indices)):
            i = i_indices[idx].item()
            j = j_indices[idx].item()
            # For both (i,j) and (j,i) permutations, use average weight
            perm_ij = pair_to_perm.get((i, j))
            perm_ji = pair_to_perm.get((j, i))
            
            if perm_ij is not None:
                weight = coatt_pred[0, perm_ij].item()
            elif perm_ji is not None:
                weight = coatt_pred[0, perm_ji].item()
            else:
                weight = 0.0
            
            edges.append((i, j, {'weight': weight}))
        
        # Build graph and detect communities
        G = nx.Graph()
        G.add_edges_from(edges)
        partition = community.best_partition(G, weight='weight', resolution=res)
        
        # Vectorize group assignment
        partition_array = np.zeros(num_people, dtype=np.int32)
        for node, comm_id in partition.items():
            partition_array[node] = comm_id
        
        # Create groups using numpy
        unique_groups = np.unique(partition_array)
        coatt_level_pred = []
        
        for g_id in unique_groups:
            members = np.where(partition_array == g_id)[0]
            if len(members) > 1:
                coatt_level_pred_grp = torch.zeros(num_people, device=coatt_level_gt.device)
                coatt_level_pred_grp[members] = 1
                coatt_level_pred.append(coatt_level_pred_grp)
        
        # Format output
        if len(coatt_level_pred) == 0:
            # coatt_level_pred = torch.zeros((1, num_people), device=coatt_level_gt.device)
            coatt_level_pred = torch.zeros((1, 1, num_people), device=coatt_level_gt.device)
        else:
            # coatt_level_pred = torch.stack(coatt_level_pred, dim=0)
            coatt_level_pred = torch.stack(coatt_level_pred, dim=0).unsqueeze(0)
        
        return coatt_level_pred

    def post_process_coatt_pred_original(self, gaze_pts_pred: list, inout_pred: list, dist_thr: float) -> list:
        """Original (non-optimized) version for group detection based on gaze point distances"""
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

    def post_process_coatt_pred(self, gaze_pts_pred: list, inout_pred: list, dist_thr: float) -> list:
        """Vectorized version for faster group detection based on gaze point distances"""
        coatt_level_pred = []
        inout_thr = 0.5
        
        for idx in range(len(gaze_pts_pred)):
            gaze_pt_pred = gaze_pts_pred[idx][0]  # (num_people, 2)
            inout_pred_b = inout_pred[idx][0]    # (num_people,)
            out_mask = inout_pred_b < inout_thr  # (num_people,)
            
            num_people = gaze_pt_pred.shape[0]
            device = gaze_pt_pred.device
            
            # Compute pairwise distances efficiently
            # gaze_pt_pred: (num_people, 2)
            # Expand for broadcasting: (num_people, 1, 2) - (1, num_people, 2)
            gaze_pt_pred_expanded = gaze_pt_pred.unsqueeze(0)  # (1, num_people, 2)
            gaze_pt_pred_transposed = gaze_pt_pred.unsqueeze(1)  # (num_people, 1, 2)
            
            # Pairwise distances: (num_people, num_people)
            dists = torch.norm(gaze_pt_pred_transposed - gaze_pt_pred_expanded, dim=-1)
            
            # Mask for valid connections (distance < threshold and both in-frame)
            in_mask = ~out_mask  # (num_people,)
            valid_mask = (dists < dist_thr)  # (num_people, num_people)
            # Both people must be in-frame
            valid_mask = valid_mask & in_mask.unsqueeze(0) & in_mask.unsqueeze(1)
            
            # Greedy group assignment
            assigned = torch.zeros(num_people, dtype=torch.bool, device=device)
            coatt_level_pred_sample = []
            
            for pi in range(num_people):
                if assigned[pi] or out_mask[pi]:
                    continue
                
                # Find all people connected to person pi
                connected = valid_mask[pi]
                connected[pi] = True  # Include self
                
                # BFS to find all connected people (group expansion)
                group = connected.clone()
                queue = torch.where(connected)[0]
                
                for qidx in queue:
                    # Expand group through transitive connections
                    group = group | valid_mask[qidx]
                
                group_indices = torch.where(group)[0]
                
                # Only create group if size > 1
                if len(group_indices) > 1:
                    coatt_level_pred_grp = torch.zeros(num_people, device=device)
                    coatt_level_pred_grp[group_indices] = 1
                    coatt_level_pred_sample.append(coatt_level_pred_grp)
                    assigned[group_indices] = True
            
            # Format output
            if len(coatt_level_pred_sample) == 0:
                coatt_level_pred_sample = torch.zeros((1, 1, num_people), device=device)
            else:
                coatt_level_pred_sample = torch.stack(coatt_level_pred_sample, dim=0).unsqueeze(0)
            
            coatt_level_pred.append(coatt_level_pred_sample)
        
        return coatt_level_pred
    
    # combine heatmaps of each person attention based on group prediction
    def generate_coatt_hm_from_level(self, coatt_level_pred: torch.Tensor, hm_pred: torch.Tensor) -> torch.Tensor:
        '''
        coatt_level_pred: (batch_size, token_num, people_num)
        hm_pred: (batch_size, num_people, img_h, img_w)
        '''

        batch_size, token_num, people_num = coatt_level_pred.shape
        batch_size, people_num, img_h, img_w = hm_pred.shape

        coatt_level_pred = coatt_level_pred.view(batch_size*token_num, people_num, 1, 1)
        hm_pred = hm_pred.view(batch_size, people_num, img_h, img_w)
        coatt_hm_pred = torch.sum(hm_pred * coatt_level_pred, dim=1)
        coatt_hm_pred = coatt_hm_pred.view(batch_size, token_num, img_h, img_w)

        return coatt_hm_pred

    def calc_group_ap(self):
        coatt_level_gt = [output['coatt_level_gt'] for output in self.test_step_outputs]
        # coatt_hm_pred = [output['coatt_hm_pred'] for output in self.test_step_outputs]
        # coatt_hm_gt = [output['coatt_hm_gt'] for output in self.test_step_outputs]
        coatt_hm_pred_pk_cd = [output['coatt_hm_pred_pk_cd'] for output in self.test_step_outputs]
        coatt_hm_pred_pk_val = [output['coatt_hm_pred_pk_val'] for output in self.test_step_outputs]
        coatt_hm_gt_pk_cd = [output['coatt_hm_gt_pk_cd'] for output in self.test_step_outputs]
        coatt_hm_gt_pk_val = [output['coatt_hm_gt_pk_val'] for output in self.test_step_outputs]
        inout_pred_w_pad = [torch.sigmoid(output['inout_pred']) for output in self.test_step_outputs]
        coatt_pred = [output['coatt_pred'] for output in self.test_step_outputs]
        hm_pred = [output['hm_pred'] for output in self.test_step_outputs]

        # remove padding in inout prediction
        head_bboxes = [output['head_bboxes'] for output in self.test_step_outputs]
        inout_pred = []
        for idx in range(len(inout_pred_w_pad)):
            head_bboxes_sample = head_bboxes[idx]
            inout_pred_w_pad_sample = inout_pred_w_pad[idx]
            pad_mask = torch.sum(head_bboxes_sample, dim=-1) <= 0
            inout_pred_w_pad_sample[pad_mask] = 0.0
            inout_pred.append(inout_pred_w_pad_sample)

            # remove padding in co-attention prediction
            head_pad_mask = torch.sum(head_bboxes[idx], dim=-1) <= 0
            indices = torch.tensor(list(itertools.permutations(torch.arange(head_bboxes_sample.shape[1]), 2)))
            for ind in range(indices.shape[0]):
                i, j = indices[ind]
                coatt_pred[idx][:, ind] = torch.where(head_pad_mask[:, i] | head_pad_mask[:, j], 0.001, coatt_pred[idx][:, ind])

        # thresholds for group-level prediction
        coatt_level_thresh_list = [round(0.1 * i, 1) for i in range(1, 10, 2)]

        # thresholds for edge-based group detection
        edge_res_list = [round(0.1 * i, 1) for i in range(10, 15, 1)]

        # thresholds for post-processing
        hm_dist_thr_pp_list = [0.05, 0.1, 0.15, 0.2, 0.3, 100.0]

        # evaluation by varying different group IoU thresholds
        group_iou_thr_list = [0.5, 0.75, 1.0]
        hm_dist_thr_list = [0.05, 0.1, 0.2, 100.0]
        
        # calculate group AP
        # for group_iou_thr in group_iou_thr_list:
            # for hm_dist_thr in hm_dist_thr_list:
        for (group_iou_thr, hm_dist_thr) in tqdm(list(itertools.product(group_iou_thr_list, hm_dist_thr_list))):
            # group_ap = GroupAP(iou_thresh=group_iou_thr, hm_thresh=hm_dist_thr)
            group_ap = GroupAPFast(iou_thresh=group_iou_thr, hm_thresh=hm_dist_thr)

            # evaluation by using our group-level prediction
            start = time.time()
            for co_lev_thr in coatt_level_thresh_list:
                # print(f"Calculating group AP for group IoU threshold {group_iou_thresh}, co-attention level threshold {co_lev_thr}...")
                coatt_level_pred = [(torch.sigmoid(output['coatt_level_pred'])>co_lev_thr).int() for output in self.test_step_outputs]
                # ret_metrics = group_ap.compute(coatt_level_pred, coatt_level_gt, coatt_hm_pred, coatt_hm_gt)
                ret_metrics = group_ap.compute(coatt_level_pred, coatt_level_gt, 
                                                coatt_hm_pred_pk_cd, coatt_hm_gt_pk_cd,
                                                coatt_hm_pred_pk_val, coatt_hm_gt_pk_val)
                self.log(f"metric/test/coatt_ap_grp_{group_iou_thr}_{hm_dist_thr}_{co_lev_thr}", ret_metrics['ap'], batch_size=1, prog_bar=True, on_step=False, on_epoch=True)
                # self.log(f"metric/test/coatt_dist_grp_{group_iou_thr}_{hm_dist_thr}_{co_lev_thr}", ret_metrics['dist'], batch_size=1, prog_bar=True, on_step=False, on_epoch=True)

            # # evaluation by using edge-based group detection
            for res in edge_res_list:
                coatt_level_pred_pairs = []
                coatt_hm_pred_pairs = []
                coatt_hm_pred_pairs_pk_cd = []
                coatt_hm_pred_pairs_pk_val = []
                for idx in range(len(coatt_pred)):
                    coatt_level_pred_pairs_sample = self.edge_base_grp_detection(coatt_pred[idx], coatt_level_gt[idx], inout_pred[idx], res)
                    coatt_level_pred_pairs.append(coatt_level_pred_pairs_sample)
                    coatt_hm_pred_pairs_sample = self.generate_coatt_hm_from_level(coatt_level_pred_pairs[idx], hm_pred[idx])
                    coatt_hm_pred_pairs.append(coatt_hm_pred_pairs_sample)
                    coatt_hm_pred_pairs_pk_cd_sample = spatial_argmax2d(coatt_hm_pred_pairs_sample.reshape(-1, coatt_hm_pred_pairs_sample.shape[2], coatt_hm_pred_pairs_sample.shape[3]), normalize=True).view(1, -1, 2)
                    coatt_hm_pred_pairs_pk_cd.append(coatt_hm_pred_pairs_pk_cd_sample)
                    coatt_hm_pred_pairs_pk_val_sample = torch.gather(coatt_hm_pred_pairs_sample.reshape(-1, coatt_hm_pred_pairs_sample.shape[2]*coatt_hm_pred_pairs_sample.shape[3]), 1, (coatt_hm_pred_pairs_pk_cd_sample[..., 0]*coatt_hm_pred_pairs_sample.shape[3] + coatt_hm_pred_pairs_pk_cd_sample[..., 1]).long())
                    coatt_hm_pred_pairs_pk_val.append(coatt_hm_pred_pairs_pk_val_sample)
                # ret_metrics = group_ap.compute(coatt_level_pred_pairs, coatt_level_gt, coatt_hm_pred_pairs, coatt_hm_gt)
                ret_metrics = group_ap.compute(coatt_level_pred_pairs, coatt_level_gt, 
                                                coatt_hm_pred_pairs_pk_cd, coatt_hm_gt_pk_cd,
                                                coatt_hm_pred_pairs_pk_val, coatt_hm_gt_pk_val)

                self.log(f"metric/test/coatt_ap_pairs_{group_iou_thr}_{hm_dist_thr}_{res}", ret_metrics['ap'], batch_size=1, prog_bar=True, on_step=False, on_epoch=True)
                # self.log(f"metric/test/coatt_dist_pairs_{group_iou_thr}_{hm_dist_thr}_{res}", ret_metrics['dist'], batch_size=1, prog_bar=True, on_step=False, on_epoch=True)

            # evaluation by using post-processing
            for dist_thr_pp in hm_dist_thr_pp_list:
                gaze_pts_pred = [output['gaze_pts_pred'] for output in self.test_step_outputs]
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
                # self.log(f"metric/test/coatt_dist_pp_{group_iou_thr}_{hm_dist_thr}_{dist_thr_pp}", ret_metrics['dist'], batch_size=1, prog_bar=True, on_step=False, on_epoch=True)

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

# ==================================================================================================================
#                                                 SHARINGAN MODEL                                                  #
# ==================================================================================================================
class SharinganModel(pl.LightningModule):

    def __init__(self, cfg):
        super().__init__()

        self.model = Sharingan(
            patch_size=cfg.model.sharingan.patch_size,
            token_dim=cfg.model.sharingan.token_dim,
            image_size=cfg.model.sharingan.image_size,
            gaze_feature_dim=cfg.model.sharingan.gaze_feature_dim,
            encoder_depth=cfg.model.sharingan.encoder_depth,
            encoder_num_heads=cfg.model.sharingan.encoder_num_heads,
            encoder_num_global_tokens=cfg.model.sharingan.encoder_num_global_tokens,
            encoder_mlp_ratio=cfg.model.sharingan.encoder_mlp_ratio,
            encoder_use_qkv_bias=cfg.model.sharingan.encoder_use_qkv_bias,
            encoder_drop_rate=cfg.model.sharingan.encoder_drop_rate,
            encoder_attn_drop_rate=cfg.model.sharingan.encoder_attn_drop_rate,
            encoder_drop_path_rate=cfg.model.sharingan.encoder_drop_path_rate,
            decoder_feature_dim=cfg.model.sharingan.decoder_feature_dim,
            decoder_hooks=cfg.model.sharingan.decoder_hooks,
            decoder_hidden_dims=cfg.model.sharingan.decoder_hidden_dims,
            decoder_use_bn=cfg.model.sharingan.decoder_use_bn,
        )

        self.cfg = cfg
        self.num_tranining_samples = 108955  # TODO: update this. VAT: ?, GazeFollow: 108955
        self.num_steps_in_epoch = math.ceil(self.num_tranining_samples / cfg.train.batch_size)
        self.test_step_outputs = []

        # Model weights Paths
        self.model_weights = cfg.model.weights
        self.gaze_weights = cfg.model.sharingan.gaze_weights
        self.multivit_weights = cfg.model.sharingan.multivit_weights

        # Define Metrics
        self.metrics = nn.ModuleDict({"val_dist": Distance(), "test_dist": GFTestDistance()})

        # Define Loss Function
        self.compute_loss = compute_sharingan_loss

        # Initialize Weights
        self._init_weights()

    def _init_weights(self):
        # Load weights for Multi ViT
        multivit_ckpt = torch.load(self.multivit_weights, map_location="cpu")
        image_tokenizer_weights = OrderedDict([(name.replace("input_adapters.rgb.", ""), value) for name, value in multivit_ckpt["model"].items() if "input_adapters.rgb" in name])
        self.model.image_tokenizer.load_state_dict(image_tokenizer_weights, strict=True)
        print(colored(f"Successfully loaded weights for the image tokenizer from {self.multivit_weights}.", TERM_COLOR))

        encoder_weights = OrderedDict([(name.replace("encoder.", ""), value) for name, value in multivit_ckpt["model"].items() if "encoder" in name])
        self.model.encoder.encoder.load_state_dict(encoder_weights, strict=True)
        print(colored(f"Successfully loaded weights for the ViT encoder from {self.multivit_weights}.", TERM_COLOR))

        # Load Gaze Encoder Gaze360 Pre-trained Weights
        gaze360_ckpt = torch.load(self.gaze_weights, map_location="cpu")
        gaze360_weights = OrderedDict([(name.replace("base_head.", ""), value) for name, value in gaze360_ckpt["model_state_dict"].items() if "base_head" in name])
        self.model.gaze_encoder.backbone.load_state_dict(gaze360_weights, strict=True)
        print(colored(f"Successfully loaded weights for the gaze backbone from {self.gaze_weights}.", TERM_COLOR))

        # Delete checkpoints
        del multivit_ckpt, image_tokenizer_weights, encoder_weights, gaze360_ckpt, gaze360_weights

    def _set_batchnorm_eval(self, module):
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()

    def _set_dropout_eval(self, module):
        if isinstance(module, torch.nn.modules.dropout._DropoutNd):
            module.eval()

    def freeze_module(self, module):
        for param in module.parameters():
            param.requires_grad = False

    def _freeze(self):
        if self.cfg.train.freeze.gaze_encoder:
            print(colored(f"Freezing the Gaze Encoder layers.", TERM_COLOR))
            self.freeze_module(self.model.gaze_encoder)
        if self.cfg.train.freeze.image_tokenizer:
            print(colored(f"Freezing the Image Tokenizer layers.", TERM_COLOR))
            self.freeze_module(self.model.image_tokenizer)
        if self.cfg.train.freeze.encoder:
            print(colored(f"Freezing the ViT Encoder layers.", TERM_COLOR))
            self.freeze_module(self.model.encoder)
        if self.cfg.train.freeze.gaze_decoder:
            print(colored(f"Freezing the Gaze Decoder layers.", TERM_COLOR))
            self.freeze_module(self.model.gaze_decoder)
        if self.cfg.train.freeze.inout_decoder:
            print(colored(f"Freezing the InOut Decoder layers.", TERM_COLOR))
            self.freeze_module(self.model.inout_decoder)

    def forward(self, batch):
        return self.model(batch)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.cfg.optimizer.lr, weight_decay=self.cfg.optimizer.weight_decay)

        T_0 = self.cfg.scheduler.t_0_epochs * self.num_steps_in_epoch
        T_mult = self.cfg.scheduler.t_mult
        lr_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0, T_mult=T_mult, eta_min=0)
        lr_scheduler_config = {"scheduler": lr_scheduler, "interval": "step", "frequency": 1}

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}

    def lr_scheduler_step(self, scheduler, *args, **kwargs):
        # Step schedule
        scheduler.step()

        # Warm-up Steps
        n = self.cfg.scheduler.warmup_epochs * self.num_steps_in_epoch
        if self.trainer.global_step < n:
            lr_scale = min(1.0, float(self.trainer.global_step + 1) / n)
            # optimizer
            for pg in scheduler.optimizer.param_groups:
                pg["lr"] = lr_scale * self.cfg.optimizer.lr

    def training_step(self, batch, batch_idx):
        n = len(batch["image"])
        ni = int(batch["inout"].sum().item())

        # Forward pass
        gaze_vec_pred, gaze_pt_pred, inout_pred = self(batch)
        gaze_vec_pred = gaze_vec_pred[:, -1, :]  # (b, n, 2) >> (b, 2)
        gaze_pt_pred = gaze_pt_pred[:, -1, :]  # (b, n, 2) >> (b, 2)
        inout_pred = inout_pred[:, -1, :].squeeze(1)  # (b, n, 1) >> (b, 1) >> (b,)

        # Compute loss
        loss, logs = self.compute_loss(batch["gaze_vec"], batch["gaze_pt"], batch["inout"], gaze_vec_pred, gaze_pt_pred, inout_pred)

        # Logging losses
        self.log("loss/train/heatmap", logs["heatmap_loss"], batch_size=ni, prog_bar=False, on_step=True, on_epoch=True)
        self.log("loss/train/dist", logs["dist_loss"], batch_size=ni, prog_bar=False, on_step=True, on_epoch=True)
        self.log("loss/train/angular", logs["angular_loss"], batch_size=ni, prog_bar=False, on_step=True, on_epoch=True)
        self.log("loss/train/bce", logs["bce_loss"], batch_size=n, prog_bar=False, on_step=True, on_epoch=True)
        self.log("loss/train", logs["total_loss"], batch_size=n, prog_bar=True, on_step=True, on_epoch=True)

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        n = len(batch["image"])
        ni = int(batch["inout"].sum().item())

        # Forward pass
        gaze_vec_pred, gaze_pt_pred, inout_pred = self(batch)
        gaze_vec_pred = gaze_vec_pred[:, -1, :]  # (b, n, 2) >> (b, 2)
        gaze_pt_pred = gaze_pt_pred[:, -1, :]  # (b, n, 2) >> (b, 2)
        inout_pred = inout_pred[:, -1, :].squeeze(1)  # (b, n, 1) >> (b, 1) >> (b,)

        # Compute loss
        loss, logs = self.compute_loss(batch["gaze_vec"], batch["gaze_pt"], batch["inout"], gaze_vec_pred, gaze_pt_pred, inout_pred)

        # Update metrics
        # self.metrics["val_auc"].update(gaze_heatmap_pred, gaze_heatmap, inout)
        self.metrics["val_dist"].update(gaze_pt_pred, batch["gaze_pt"], batch["inout"])

        # Logging losses
        self.log("loss/val/heatmap", logs["heatmap_loss"], batch_size=ni, prog_bar=False, on_step=False, on_epoch=True)
        self.log("loss/val/dist", logs["dist_loss"], batch_size=ni, prog_bar=False, on_step=False, on_epoch=True)
        self.log("loss/val/angular", logs["angular_loss"], batch_size=ni, prog_bar=False, on_step=True, on_epoch=True)
        self.log("loss/val/bce", logs["bce_loss"], batch_size=n, prog_bar=False, on_step=False, on_epoch=True)
        self.log("loss/val", logs["total_loss"], batch_size=n, prog_bar=True, on_step=False, on_epoch=True)

        # Logging metrics
        # self.log("metric/val/auc", self.metrics["val_auc"], batch_size=ni, prog_bar=True, on_step=False, on_epoch=True)
        self.log("metric/val/dist", self.metrics["val_dist"], batch_size=ni, prog_bar=True, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        n = len(batch["image"])
        ni = int(batch["inout"].sum().item())
        assert n == ni, f"Expected all test samples to be looking inside. Got {n} samples, {ni} of which are looking inside."

        # Forward pass
        gaze_vec_pred, gaze_pt_pred, inout_pred = self(batch)
        gaze_vec_pred = gaze_vec_pred[:, -1, :]  # (b, n, 2) >> (b, 2)
        gaze_pt_pred = gaze_pt_pred[:, -1, :]  # (b, n, 2) >> (b, 2)
        inout_pred = inout_pred[:, -1, :].squeeze(1)  # (b, n, 1) >> (b, 1) >> (b,)

        # Update metrics
        # test_auc = self.metrics["test_auc"](gaze_heatmap_pred, gaze_pt)
        test_dist_to_avg, test_avg_dist, test_min_dist = self.metrics["test_dist"](gaze_pt_pred, batch["gaze_pt"])

        # Log metrics
        # self.log("metric/test/auc", test_auc, batch_size=n, prog_bar=True, on_step=False, on_epoch=True)
        self.log("metric/test/dist_to_avg", test_dist_to_avg, batch_size=n, prog_bar=True, on_step=False, on_epoch=True)
        self.log("metric/test/avg_dist", test_avg_dist, batch_size=n, prog_bar=True, on_step=False, on_epoch=True)
        self.log("metric/test/min_dist", test_min_dist, batch_size=n, prog_bar=True, on_step=False, on_epoch=True)

        # Build output dict
        output = {"gp_pred": gaze_pt_pred, "inout_pred": inout_pred, "gp_gt": batch["gaze_pt"], "inout_gt": batch["inout"], "path": batch["path"]}
        self.test_step_outputs.append(output)

    def on_test_epoch_end(self):
        # Reset metrics
        self.metrics["test_dist"].reset()
        # self.metrics["test_auc"].reset()

        # Save test predictions
        self._save_predictions(self.test_step_outputs)

    def _save_predictions(self, outputs):
        with open("./test-predictions.pickle", "wb") as file:
            pickle.dump(outputs, file)


# ==================================================================================================================
#                                                   CHONG MODEL                                                    #
# ==================================================================================================================
class ChongModel(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()

        self.model = ChongNet()

        self.cfg = cfg
        self.num_tranining_samples = 108955  # TODO: update this. VAT: ?, GazeFollow: 108955
        self.num_steps_in_epoch = math.ceil(self.num_tranining_samples / cfg.train.batch_size)
        self.test_step_outputs = []

        # Model Weights Path
        self.model_weights = "/idiap/temp/stafasca/projects/rinnegan/weights/chong_initial_weights_spatial_model.pt"

        # Define Metrics
        self.metrics = nn.ModuleDict({"val_dist": Distance(), "val_auc": AUC(), "test_dist": GFTestDistance(), "test_auc": GFTestAUC()})

        # Define Loss Function
        self.compute_loss = compute_chong_loss

        # Initialize Weights
        self._init_weights()

    def _init_weights(self):
        pretrained_dict = torch.load(self.model_weights, map_location="cpu")["model"]
        model_dict = self.model.state_dict()
        model_dict.update(pretrained_dict)
        self.model.load_state_dict(model_dict)
        print(colored(f"Successfully loaded initial weights from {self.model_weights}.", TERM_COLOR))
        del pretrained_dict, model_dict

    def forward(self, image, head_mask, head):
        return self.model(image, head_mask, head)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.cfg.optimizer.lr)
        T_0 = self.cfg.scheduler.t_0_epochs * self.num_steps_in_epoch
        T_mult = self.cfg.scheduler.t_mult
        lr_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0, T_mult=T_mult, eta_min=0)
        lr_scheduler_config = {"scheduler": lr_scheduler, "interval": "step", "frequency": 1}
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}

    def training_step(self, batch, batch_idx):
        image = batch["image"]
        head_mask = batch["head_masks"].squeeze(1)
        head = batch["heads"].squeeze(1)
        gaze_heatmap = batch["gaze_heatmap"]
        inout = batch["inout"]
        n = len(image)
        ni = int(inout.sum().item())

        # Forward pass
        gaze_heatmap_pred, attmap, inout_pred = self(image, head_mask, head)
        gaze_heatmap_pred = gaze_heatmap_pred.squeeze(1)  # (b, 1, 64, 64) >> (b, 64, 64)
        inout_pred = inout_pred.squeeze(1)  # (b, 1) >> (b,)

        # Compute loss
        loss, logs = self.compute_loss(gaze_heatmap, inout, gaze_heatmap_pred, inout_pred)

        # Logging losses
        self.log("loss/train/heatmap", logs["heatmap_loss"], batch_size=ni, prog_bar=False, on_step=True, on_epoch=True)
        self.log("loss/train/bce", logs["bce_loss"], batch_size=n, prog_bar=False, on_step=True, on_epoch=True)
        self.log("loss/train", logs["total_loss"], batch_size=n, prog_bar=True, on_step=True, on_epoch=True)

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        image = batch["image"]
        head_mask = batch["head_masks"].squeeze(1)
        head = batch["heads"].squeeze(1)
        gaze_pt = batch["gaze_pt"]
        gaze_heatmap = batch["gaze_heatmap"]
        inout = batch["inout"]
        n = len(image)
        ni = int(inout.sum().item())

        # Forward pass
        gaze_heatmap_pred, attmap, inout_pred = self(image, head_mask, head)
        gaze_heatmap_pred = gaze_heatmap_pred.squeeze(1)  # (b, 1, 64, 64) >> (b, 64, 64)
        inout_pred = inout_pred.squeeze(1)  # (b, 1) >> (b,)
        gaze_pt_pred = spatial_argmax2d(gaze_heatmap_pred, normalize=True)  # (b, 2)

        # Compute loss
        loss, logs = self.compute_loss(gaze_heatmap, inout, gaze_heatmap_pred, inout_pred)

        # Update metrics
        self.metrics["val_auc"].update(gaze_heatmap_pred, gaze_heatmap, inout)
        self.metrics["val_dist"].update(gaze_pt_pred, gaze_pt, inout)

        # Logging losses
        self.log("loss/val/heatmap", logs["heatmap_loss"], batch_size=ni, prog_bar=False, on_step=False, on_epoch=True)
        self.log("loss/val/bce", logs["bce_loss"], batch_size=n, prog_bar=False, on_step=False, on_epoch=True)
        self.log("loss/val", logs["total_loss"], batch_size=n, prog_bar=True, on_step=False, on_epoch=True)

        # Logging metrics
        self.log("metric/val/auc", self.metrics["val_auc"], batch_size=ni, prog_bar=True, on_step=False, on_epoch=True)
        self.log("metric/val/dist", self.metrics["val_dist"], batch_size=ni, prog_bar=True, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        image = batch["image"]
        head_mask = batch["head_masks"].squeeze(1)
        head = batch["heads"].squeeze(1)
        gaze_pt = batch["gaze_pt"]
        gaze_heatmap = batch["gaze_heatmap"]
        inout = batch["inout"]
        n = len(image)
        ni = int(inout.sum().item())
        assert n == ni, f"Expected all test samples to be looking inside. Got {n} samples, {ni} of which are looking inside."

        # Forward pass
        gaze_heatmap_pred, attmap, inout_pred = self(image, head_mask, head)
        gaze_heatmap_pred = gaze_heatmap_pred.squeeze(1)  # (b, 1, 64, 64) >> (b, 64, 64)
        inout_pred = inout_pred.squeeze(1)  # (b, 1) >> (b,)
        gaze_pt_pred = spatial_argmax2d(gaze_heatmap_pred, normalize=True)  # (b, 2)

        # Update metrics
        test_auc = self.metrics["test_auc"](gaze_heatmap_pred, gaze_pt)
        test_dist_to_avg, test_avg_dist, test_min_dist = self.metrics["test_dist"](gaze_pt_pred, gaze_pt)

        # Log metrics
        self.log("metric/test/auc", test_auc, batch_size=n, prog_bar=True, on_step=False, on_epoch=True)
        self.log("metric/test/dist_to_avg", test_dist_to_avg, batch_size=n, prog_bar=True, on_step=False, on_epoch=True)
        self.log("metric/test/avg_dist", test_avg_dist, batch_size=n, prog_bar=True, on_step=False, on_epoch=True)
        self.log("metric/test/min_dist", test_min_dist, batch_size=n, prog_bar=True, on_step=False, on_epoch=True)

        # Build output dict
        output = {"gp_pred": gaze_pt_pred, "inout_pred": inout_pred, "gp_gt": gaze_pt, "inout_gt": inout, "path": batch["path"]}
        self.test_step_outputs.append(output)

    def on_test_epoch_end(self):
        # Reset metrics
        self.metrics["test_dist"].reset()
        self.metrics["test_auc"].reset()

        # Save test predictions
        self._save_predictions(self.test_step_outputs)

    def _save_predictions(self, outputs):
        with open("./test-predictions.pickle", "wb") as file:
            pickle.dump(outputs, file)


# ==================================================================================================================
#                                                    NORA MODEL                                                    #
# ==================================================================================================================
class NoraModel(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()

        self.model = NoraNet()

        self.cfg = cfg
        self.num_tranining_samples = 108955  # TODO: update this. VAT: ?, GazeFollow: 108955
        self.num_steps_in_epoch = math.ceil(self.num_tranining_samples / cfg.train.batch_size)
        self.test_step_outputs = []

        # Model Weights Path
        # self.model_weights = "/idiap/temp/stafasca/projects/rinnegan/weights/chong_initial_weights_spatial_model.pt"
        self.model_weights = "/idiap/temp/stafasca/projects/rinnegan/weights/chong_model_demo.pt"

        # Define Metrics
        self.metrics = nn.ModuleDict({"val_dist": Distance(), "val_auc": AUC(), "test_dist": GFTestDistance(), "test_auc": GFTestAUC()})

        # Define Loss Function
        self.compute_loss = compute_chong_loss

        # Initialize Weights
        self._init_weights()

        # Freeze Gaze360
        self.freeze_module(self.model.gaze360)

    def freeze_module(self, module):
        for param in module.parameters():
            param.requires_grad = False

    def _init_weights(self):
        checkpoint = torch.load(self.model_weights, map_location="cpu")["model"]
        self.model.load_state_dict(checkpoint, strict=False)
        print(colored(f"Successfully loaded initial weights from {self.model_weights}.", TERM_COLOR))
        del checkpoint

    def forward(self, sample):
        return self.model(sample)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.cfg.optimizer.lr)
        T_0 = self.cfg.scheduler.t_0_epochs * self.num_steps_in_epoch
        T_mult = self.cfg.scheduler.t_mult
        lr_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0, T_mult=T_mult, eta_min=0)
        lr_scheduler_config = {"scheduler": lr_scheduler, "interval": "step", "frequency": 1}
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}

    def lr_scheduler_step(self, scheduler, optimizer_idx):
        # Step schedule
        scheduler.step()

        # Warm-up Steps
        n = self.cfg.scheduler.warmup_epochs * self.num_steps_in_epoch
        if self.trainer.global_step < n:
            lr_scale = min(1.0, float(self.trainer.global_step + 1) / n)
            # optimizer
            for pg in scheduler.optimizer.param_groups:
                pg["lr"] = lr_scale * self.cfg.optimizer.lr

    def on_train_epoch_start(self):
        self.model.gaze360.eval()

    def training_step(self, batch, batch_idx):
        image = batch["image"]
        head_mask = batch["head_masks"].squeeze(1)
        head = batch["heads"].squeeze(1)
        gaze_heatmap = batch["gaze_heatmap"]
        inout = batch["inout"]
        n = len(image)
        ni = int(inout.sum().item())

        # Forward pass
        gaze_heatmap_pred = self(batch)
        inout_pred = torch.ones_like(batch["inout"])
        gaze_heatmap_pred = gaze_heatmap_pred.squeeze(1)  # (b, 1, 64, 64) >> (b, 64, 64)

        # Compute loss
        loss, logs = self.compute_loss(gaze_heatmap, inout, gaze_heatmap_pred, inout_pred)

        # Logging losses
        self.log("loss/train/heatmap", logs["heatmap_loss"], batch_size=ni, prog_bar=False, on_step=True, on_epoch=True)
        self.log("loss/train/bce", logs["bce_loss"], batch_size=n, prog_bar=False, on_step=True, on_epoch=True)
        self.log("loss/train", logs["total_loss"], batch_size=n, prog_bar=True, on_step=True, on_epoch=True)

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        image = batch["image"]
        head_mask = batch["head_masks"].squeeze(1)
        head = batch["heads"].squeeze(1)
        gaze_pt = batch["gaze_pt"]
        gaze_heatmap = batch["gaze_heatmap"]
        inout = batch["inout"]
        n = len(image)
        ni = int(inout.sum().item())

        # Forward pass
        gaze_heatmap_pred = self(batch)
        inout_pred = torch.ones_like(batch["inout"])
        gaze_heatmap_pred = gaze_heatmap_pred.squeeze(1)  # (b, 1, 64, 64) >> (b, 64, 64)
        gaze_pt_pred = spatial_argmax2d(gaze_heatmap_pred, normalize=True)  # (b, 2)

        # Compute loss
        loss, logs = self.compute_loss(gaze_heatmap, inout, gaze_heatmap_pred, inout_pred)

        # Update metrics
        self.metrics["val_auc"].update(gaze_heatmap_pred, gaze_heatmap, inout)
        self.metrics["val_dist"].update(gaze_pt_pred, gaze_pt, inout)

        # Logging losses
        self.log("loss/val/heatmap", logs["heatmap_loss"], batch_size=ni, prog_bar=False, on_step=False, on_epoch=True)
        self.log("loss/val/bce", logs["bce_loss"], batch_size=n, prog_bar=False, on_step=False, on_epoch=True)
        self.log("loss/val", logs["total_loss"], batch_size=n, prog_bar=True, on_step=False, on_epoch=True)

        # Logging metrics
        self.log("metric/val/auc", self.metrics["val_auc"], batch_size=ni, prog_bar=True, on_step=False, on_epoch=True)
        self.log("metric/val/dist", self.metrics["val_dist"], batch_size=ni, prog_bar=True, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        image = batch["image"]
        head_mask = batch["head_masks"].squeeze(1)
        head = batch["heads"].squeeze(1)
        gaze_pt = batch["gaze_pt"]
        gaze_heatmap = batch["gaze_heatmap"]
        inout = batch["inout"]
        n = len(image)
        ni = int(inout.sum().item())
        assert n == ni, f"Expected all test samples to be looking inside. Got {n} samples, {ni} of which are looking inside."

        # Forward pass
        gaze_heatmap_pred = self(batch)
        inout_pred = torch.ones_like(batch["inout"])
        gaze_heatmap_pred = gaze_heatmap_pred.squeeze(1)  # (b, 1, 64, 64) >> (b, 64, 64)
        gaze_pt_pred = spatial_argmax2d(gaze_heatmap_pred, normalize=True)  # (b, 2)

        # Update metrics
        test_auc = self.metrics["test_auc"](gaze_heatmap_pred, gaze_pt)
        test_dist_to_avg, test_avg_dist, test_min_dist = self.metrics["test_dist"](gaze_pt_pred, gaze_pt)

        # Log metrics
        self.log("metric/test/auc", test_auc, batch_size=n, prog_bar=True, on_step=False, on_epoch=True)
        self.log("metric/test/dist_to_avg", test_dist_to_avg, batch_size=n, prog_bar=True, on_step=False, on_epoch=True)
        self.log("metric/test/avg_dist", test_avg_dist, batch_size=n, prog_bar=True, on_step=False, on_epoch=True)
        self.log("metric/test/min_dist", test_min_dist, batch_size=n, prog_bar=True, on_step=False, on_epoch=True)

        # Build output dict
        output = {"gp_pred": gaze_pt_pred, "inout_pred": inout_pred, "gp_gt": gaze_pt, "inout_gt": inout, "path": batch["path"]}
        self.test_step_outputs.append(output)

    def on_test_epoch_end(self):
        # Reset metrics
        self.metrics["test_dist"].reset()
        self.metrics["test_auc"].reset()

        # Save test predictions
        self._save_predictions(self.test_step_outputs)

    def _save_predictions(self, outputs):
        with open("./test-predictions.pickle", "wb") as file:
            pickle.dump(outputs, file)


# ==================================================================================================================
#                                                 BASELINE MODEL                                                   #
# ==================================================================================================================
class BaselineModel(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()

        self.model = GazeBaseline()

        self.cfg = cfg
        self.num_tranining_samples = 108955
        self.num_steps_in_epoch = math.ceil(self.num_tranining_samples / cfg.train.batch_size)
        self.test_step_outputs = []

        # Define Metrics
        self.metrics = nn.ModuleDict({"val_dist": Distance(), "val_auc": AUC(), "test_dist": GFTestDistance(), "test_auc": GFTestAUC()})

        # Define Loss Function
        self.compute_loss = compute_chong_loss

        # Freeze Gaze360
        self.freeze_module(self.model.gaze360)

    def freeze_module(self, module):
        for param in module.parameters():
            param.requires_grad = False

    def forward(self, sample):
        return self.model(sample)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.cfg.optimizer.lr)
        T_0 = self.cfg.scheduler.t_0_epochs * self.num_steps_in_epoch
        T_mult = self.cfg.scheduler.t_mult
        lr_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0, T_mult=T_mult, eta_min=0)
        lr_scheduler_config = {"scheduler": lr_scheduler, "interval": "step", "frequency": 1}
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}

    def lr_scheduler_step(self, scheduler, optimizer_idx):
        # Step schedule
        scheduler.step()

        # Warm-up Steps
        n = self.cfg.scheduler.warmup_epochs * self.num_steps_in_epoch
        if self.trainer.global_step < n:
            lr_scale = min(1.0, float(self.trainer.global_step + 1) / n)
            # optimizer
            for pg in scheduler.optimizer.param_groups:
                pg["lr"] = lr_scale * self.cfg.optimizer.lr

    def on_train_epoch_start(self):
        self.model.gaze360.eval()

    def training_step(self, batch, batch_idx):
        n = len(batch["image"])
        ni = int(batch["inout"].sum().item())

        # Forward pass
        gaze_heatmap_pred = self(batch)
        inout_pred = torch.ones_like(batch["inout"])
        gaze_heatmap_pred = gaze_heatmap_pred.squeeze(1)  # (b, 1, 64, 64) >> (b, 64, 64)

        # Compute loss
        loss, logs = self.compute_loss(batch["gaze_heatmap"], batch["inout"], gaze_heatmap_pred, inout_pred)

        # Logging losses
        self.log("loss/train/heatmap", logs["heatmap_loss"], batch_size=ni, prog_bar=False, on_step=True, on_epoch=True)
        self.log("loss/train/bce", logs["bce_loss"], batch_size=n, prog_bar=False, on_step=True, on_epoch=True)
        self.log("loss/train", logs["total_loss"], batch_size=n, prog_bar=True, on_step=True, on_epoch=True)

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        n = len(batch["image"])
        ni = int(batch["inout"].sum().item())

        # Forward pass
        gaze_heatmap_pred = self(batch)
        inout_pred = torch.ones_like(batch["inout"])
        gaze_heatmap_pred = gaze_heatmap_pred.squeeze(1)  # (b, 1, 64, 64) >> (b, 64, 64)
        gaze_pt_pred = spatial_argmax2d(gaze_heatmap_pred, normalize=True)  # (b, 2)

        # Compute loss
        loss, logs = self.compute_loss(batch["gaze_heatmap"], batch["inout"], gaze_heatmap_pred, inout_pred)

        # Update metrics
        self.metrics["val_auc"].update(gaze_heatmap_pred, batch["gaze_heatmap"], batch["inout"])
        self.metrics["val_dist"].update(gaze_pt_pred, batch["gaze_pt"], batch["inout"])

        # Logging losses
        self.log("loss/val/heatmap", logs["heatmap_loss"], batch_size=ni, prog_bar=False, on_step=False, on_epoch=True)
        self.log("loss/val/bce", logs["bce_loss"], batch_size=n, prog_bar=False, on_step=False, on_epoch=True)
        self.log("loss/val", logs["total_loss"], batch_size=n, prog_bar=True, on_step=False, on_epoch=True)

        # Logging metrics
        self.log("metric/val/auc", self.metrics["val_auc"], batch_size=ni, prog_bar=True, on_step=False, on_epoch=True)
        self.log("metric/val/dist", self.metrics["val_dist"], batch_size=ni, prog_bar=True, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        n = len(batch["image"])
        ni = int(batch["inout"].sum().item())
        assert n == ni, f"Expected all test samples to be looking inside. Got {n} samples, {ni} of which are looking inside."

        # Forward pass
        gaze_heatmap_pred = self(batch)
        inout_pred = torch.ones_like(batch["inout"])
        gaze_heatmap_pred = gaze_heatmap_pred.squeeze(1)  # (b, 1, 64, 64) >> (b, 64, 64)
        gaze_pt_pred = spatial_argmax2d(gaze_heatmap_pred, normalize=True)  # (b, 2)

        # Update metrics
        test_auc = self.metrics["test_auc"](gaze_heatmap_pred, batch["gaze_pt"])
        test_dist_to_avg, test_avg_dist, test_min_dist = self.metrics["test_dist"](gaze_pt_pred, batch["gaze_pt"])

        # Log metrics
        self.log("metric/test/auc", test_auc, batch_size=n, prog_bar=True, on_step=False, on_epoch=True)
        self.log("metric/test/dist_to_avg", test_dist_to_avg, batch_size=n, prog_bar=True, on_step=False, on_epoch=True)
        self.log("metric/test/avg_dist", test_avg_dist, batch_size=n, prog_bar=True, on_step=False, on_epoch=True)
        self.log("metric/test/min_dist", test_min_dist, batch_size=n, prog_bar=True, on_step=False, on_epoch=True)

        # Build output dict
        output = {"gp_pred": gaze_pt_pred, "inout_pred": inout_pred, "gp_gt": batch["gaze_pt"], "inout_gt": batch["inout"], "path": batch["path"]}
        self.test_step_outputs.append(output)

    def on_test_epoch_end(self):
        # Reset metrics
        self.metrics["test_dist"].reset()
        self.metrics["test_auc"].reset()

        # Save test predictions
        self._save_predictions(self.test_step_outputs)

    def _save_predictions(self, outputs):
        with open("./test-predictions.pickle", "wb") as file:
            pickle.dump(outputs, file)
