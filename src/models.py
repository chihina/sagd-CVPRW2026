import math
import pickle
from collections import OrderedDict
from typing import Dict, Union
import itertools
import sys

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

from src.losses import compute_chong_loss, compute_sharingan_loss, compute_rinnegan_loss, compute_social_loss, compute_interact_loss, compute_inout_loss, compute_coatt_loss
from src.metrics import AUC, Distance, GFTestAUC, GFTestDistance
from src.networks.chong import ChongNet, GazeBaseline, NoraNet
# from src.networks.rinnegan import Rinnegan
# from src.networks.rinnegan_multivit import MultiViTRinnegan
from src.networks.sharingan import Sharingan
from src.networks.sharingan_social import Sharingan_social
from src.networks.interact_net_temporal import InteractNet
from src.networks.geom_gaze import GeomGaze
from src.utils import spatial_argmax2d, id_to_pairwise_coatt, id_to_pairwise_lah, id_to_pairwise_laeo

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
        if self.cfg.train.freeze.gaze_decoder:
            print(colored(f"Freezing the Gaze Decoder layers.", TERM_COLOR))
            self.freeze_module(self.model.gaze_decoder)
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
            lr_scheduler = MultiStepLR(optimizer, milestones=[10,11,12,13,14,15,16], gamma=0.5)
#             lr_scheduler = StepLR(optimizer, step_size=self.cfg.scheduler.t_0_epochs, gamma=0.1)
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
        if self.cfg.train.freeze.gaze_decoder:
            self.model.gaze_decoder.apply(self._set_batchnorm_eval)
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
            coatt_hm_pred = out['coatt_hm']
            coatt_level_pred = out['coatt_level']
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
        # Compute social gaze loss
        if self.model_name in ['gaze_interact', 'sharingan_social']:
            # coatt_gt, coatt_mask = id_to_pairwise_coatt(batch["coatt_ids"].view(batch_size*t, -1))
            # laeo_gt, laeo_mask = id_to_pairwise_laeo(batch["laeo_ids"].view(batch_size*t, -1))
            # lah_gt, lah_mask = id_to_pairwise_lah(batch["lah_ids"].view(batch_size*t, -1))
            # loss_social, logs_social = self.compute_social_loss(lah_pred, lah_gt, lah_mask, laeo_pred, laeo_gt, laeo_mask, coatt_pred, coatt_gt, coatt_mask)
            # loss += loss_social  
        
            # Log Social Gaze Losses
            # self.log("loss/train/lah", logs_social["lah_loss"], batch_size=lah_mask.sum(), prog_bar=True, on_step=True, on_epoch=True)
            # self.log("loss/train/laeo", logs_social["laeo_loss"], batch_size=laeo_mask.sum(), prog_bar=True, on_step=True, on_epoch=True)
            # self.log("loss/train/coatt", logs_social["coatt_loss"], batch_size=coatt_mask.sum(), prog_bar=True, on_step=True, on_epoch=True)
            
            # Compute coatt loss
            coatt_hm_gt = batch['coatt_heatmaps']
            coatt_level_gt = batch['coatt_levels']
            coatt_loss, logs_coatt = self.compute_coatt_loss(coatt_hm_gt, coatt_hm_pred, coatt_level_gt, coatt_level_pred)
            loss += coatt_loss
            
            # Log Coatt Losses
            self.log("loss/train/coatt_hm", logs_coatt["coatt_hm_loss"], batch_size=ni, prog_bar=False, on_step=True, on_epoch=True)
            self.log("loss/train/coatt_level", logs_coatt["coatt_level_loss"], batch_size=ni, prog_bar=False, on_step=True, on_epoch=True)

        # Logging Distance, InOut losses
        self.log("loss/train/heatmap", logs_dist["heatmap_loss"], batch_size=ni, prog_bar=False, on_step=True, on_epoch=True)
        self.log("loss/train/dist", logs_dist["dist_loss"], batch_size=ni, prog_bar=False, on_step=True, on_epoch=True)
        self.log("loss/train/angular", logs_dist["angular_loss"], batch_size=ni, prog_bar=False, on_step=True, on_epoch=True)
        self.log("loss/train/inout", logs_dist["inout_loss"], batch_size=n, prog_bar=False, on_step=True, on_epoch=True)
        self.log("loss/train", loss.item(), batch_size=n, prog_bar=True, on_step=True, on_epoch=True)

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
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
            coatt_hm_pred = out['coatt_hm']
            coatt_level_pred = out['coatt_level']

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
        coatt_hm_pred = coatt_hm_pred[:,middle_frame_idx,:,:,:]
        coatt_level_pred = coatt_level_pred[:,middle_frame_idx,:,:]

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
        self.log("loss/val/dist", logs_dist["dist_loss"], batch_size=ni, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        self.log("loss/val/angular", logs_dist["angular_loss"], batch_size=ni, prog_bar=False, on_step=True, on_epoch=True, sync_dist=True)
        self.log("loss/val/inout", logs_dist["inout_loss"], batch_size=n, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        self.log("loss/val", loss.item(), batch_size=n, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        
        # Update dist metrics
        # self.metrics["val_auc"].update(gaze_heatmap_pred, gaze_heatmap, inout)
        self.metrics["val_dist"].update(gaze_pt_pred, batch["gaze_pts"][:,middle_frame_idx,:, :], batch["inout"][:,middle_frame_idx,:])
        # self.log("metric/val/auc", self.metrics["val_auc"], batch_size=ni, prog_bar=True, on_step=False, on_epoch=True)
        self.log("metric/val/dist", self.metrics["val_dist"], batch_size=ni, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        
        # Compute social gaze loss
        if self.model_name in ['gaze_interact', 'sharingan_social']:
            coatt_gt, coatt_mask = id_to_pairwise_coatt(batch["coatt_ids"][:,middle_frame_idx,:])
            laeo_gt, laeo_mask = id_to_pairwise_laeo(batch["laeo_ids"][:,middle_frame_idx,:])
            lah_gt, lah_mask = id_to_pairwise_lah(batch["lah_ids"][:,middle_frame_idx,:])
            # loss_social, logs_social = self.compute_social_loss(lah_pred, lah_gt, lah_mask, laeo_pred, laeo_gt, laeo_mask, coatt_pred, coatt_gt, coatt_mask)
            # loss += loss_social        
            coatt_gt = torch.where(coatt_mask, coatt_gt, torch.tensor(-1., device=coatt_mask.device)).int()
            laeo_gt = torch.where(laeo_mask, laeo_gt, torch.tensor(-1., device=laeo_mask.device)).int()
            lah_gt = torch.where(lah_mask, lah_gt, torch.tensor(-1., device=lah_mask.device)).int()
        
            # Log Social Gaze Losses
            # self.log("loss/val/lah", logs_social["lah_loss"], batch_size=lah_mask.sum(), prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
            # self.log("loss/val/laeo", logs_social["laeo_loss"], batch_size=laeo_mask.sum(), prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
            # self.log("loss/val/coatt", logs_social["coatt_loss"], batch_size=coatt_mask.sum(), prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)

            # Compute coatt loss
            coatt_hm_gt = batch['coatt_heatmaps'][:,middle_frame_idx,:,:,:]
            coatt_levels_gt = batch['coatt_levels'][:,middle_frame_idx,:,:]
            coatt_loss, logs_coatt = self.compute_coatt_loss(coatt_hm_gt, coatt_hm_pred, coatt_levels_gt, coatt_level_pred)

            # Log Coatt Losses
            loss += coatt_loss
            self.log("loss/val/coatt_hm", logs_coatt["coatt_hm_loss"], batch_size=ni, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
            self.log("loss/val/coatt_level", logs_coatt["coatt_level_loss"], batch_size=ni, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        
        # Update CoAtt Metrics
        if coatt_pred.sum()!=0:
            coatt_pred = torch.sigmoid(coatt_pred)
            if coatt_mask.sum()>0:
                self.val_coatt_auc(coatt_pred, coatt_gt)
                self.val_coatt_ap(coatt_pred, coatt_gt)

                self.log("metric/val/coatt_auc", self.val_coatt_auc, batch_size=coatt_mask.sum(), prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
                self.log("metric/val/coatt_ap", self.val_coatt_ap, batch_size=coatt_mask.sum(), prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        
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
        
    def test_step(self, batch, batch_idx):
        ni = int((batch["inout"]==1).sum().item())
#         assert n == ni, f"Expected all test samples to be looking inside. Got {n} samples, {ni} of which are looking inside."

        # Forward pass
        if self.output=='heatmap':
            out = self(batch)
            gaze_vec_pred = out['gaze_vec']
            gaze_hm_pred = out['gaze_hm']
            inout_pred = out['inout']
            lah_pred = out['lah']
            laeo_pred = out['laeo']
            coatt_pred = out['coatt']
            coatt_hm_pred = out['coatt_hm']
            coatt_level_pred = out['coatt_level']

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
        coatt_hm_pred = coatt_hm_pred[:,middle_frame_idx,:,:,:]
        coatt_level_pred = coatt_level_pred[:,middle_frame_idx,:,:]
        
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
            self.log("metric/test/auc", test_auc, batch_size=ni, prog_bar=True, on_step=False, on_epoch=True)
            self.log("metric/test/dist_to_avg", test_dist_to_avg, batch_size=ni, prog_bar=True, on_step=False, on_epoch=True)
            self.log("metric/test/avg_dist", test_avg_dist, batch_size=ni, prog_bar=True, on_step=False, on_epoch=True)
            self.log("metric/test/min_dist", test_min_dist, batch_size=ni, prog_bar=True, on_step=False, on_epoch=True)
        else:
            inout_gt = batch["inout"][:,middle_frame_idx,:]   
            # Log metrics
            if self.output=='heatmap':
                test_auc = self.metrics["test_auc"](gaze_hm_pred.reshape(batch_size*num_people, hm_h, hm_w), batch["gaze_heatmaps"][:,middle_frame_idx,:,:,:].reshape(batch_size*num_people, hm_h, hm_w), inout_gt.reshape(batch_size*num_people, -1))
                self.log("metric/test/auc", test_auc, batch_size=ni, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
            self.metrics["test_dist"].update(gaze_pt_pred, batch["gaze_pts"][:,middle_frame_idx,:, :], inout_gt)
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
                self.test_coatt_auc(coatt_pred, coatt_gt)
                self.test_coatt_ap(coatt_pred, coatt_gt)

                self.log("metric/test/coatt_auc", self.test_coatt_auc, batch_size=coatt_mask.sum(), prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
                self.log("metric/test/coatt_ap", self.test_coatt_ap, batch_size=coatt_mask.sum(), prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        
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

                self.log("metric/test/laeo_auc", self.test_laeo_auc, batch_size=laeo_mask.sum(), prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
                self.log("metric/test/laeo_ap", self.test_laeo_ap, batch_size=laeo_mask.sum(), prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        
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

                self.log("metric/test/lah_auc", self.test_lah_auc, batch_size=(lah_gt_metric!=-1).sum(), prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
                self.log("metric/test/lah_ap", self.test_lah_ap, batch_size=(lah_gt_metric!=-1).sum(), prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)

        # Build output dict
        output = {"head_bboxes": batch["head_bboxes"][:,middle_frame_idx,:, :],
                  "gp_pred": gaze_pt_pred, 
                  "gp_gt": batch["gaze_pts"][:,middle_frame_idx,:, :],
                  "gv_pred": gaze_vec_pred, 
                  "gv_gt": batch["gaze_vecs"][:,middle_frame_idx,:, :],
                #   # optionally save gaze heatmaps
                #   "hm_pred": gaze_hm_pred,
                #   "hm_gt": batch["gaze_heatmaps"][:,middle_frame_idx,:,:,:],
                  "inout_gt": inout_gt, 
                  "path": batch["path"],
                  "inout_pred": inout_pred,
                  "coatt_pred": coatt_pred,
                  "coatt_hm_pred": coatt_hm_pred,
                  "coatt_level_pred": coatt_level_pred,
                  "laeo_pred": laeo_pred,
                  "lah_pred": lah_pred,
                  "coatt_gt": coatt_gt,
                  "laeo_gt": laeo_gt,
                  "laeo_ids": batch['laeo_ids'][:,middle_frame_idx,:],
                  "lah_gt": lah_gt,
                  "dataset": batch['dataset'],
                  "is_child": batch['is_child'][:,middle_frame_idx,:],
                  "speaking": batch['speaking'][:,middle_frame_idx,:],
                  "num_valid_people": batch['num_valid_people']}
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
