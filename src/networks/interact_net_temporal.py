import math
import sys
from typing import Dict, List, Optional, Tuple, Union
import itertools

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import torchvision.transforms.functional as TF

from src.utils import pair, build_2d_sincos_posemb
from src.networks.adaptor_modules import InteractionBlock, CoAttCrossAttention

# ==================================================================================================================== #
#                                                INTERACT-NET ARCHITECTURE                                                #
# ==================================================================================================================== #
class InteractNet(nn.Module):
    def __init__(self,
                 cfg: Dict,
                 patch_size: int = 16,
                 token_dim: int = 768,
                 image_size: int = 224,
                 gaze_feature_dim: int = 512,
                 encoder_depth: int = 12,
                 encoder_num_heads: int = 12,
                 encoder_num_global_tokens: int = 1,
                 encoder_mlp_ratio: float = 4.0,
                 encoder_use_qkv_bias: bool = True,
                 encoder_drop_rate: float = 0.0,
                 encoder_attn_drop_rate: float = 0.0,
                 encoder_drop_path_rate: float = 0.0,
                 decoder_feature_dim: int = 256,
                 decoder_hooks: list = [2, 5, 8, 11],
                 decoder_hidden_dims: list = [96, 192, 384, 768],
                 decoder_use_bn: bool = False,
                 proj_feature_dim: int = 128,
                 vlm_dim: int = 24,
                 num_coatt: int = 30,
                 temporal_context: int=2,
                 hm_size = (64,64),    # width, height
                 output='heatmap'):
        
        super().__init__()
        
        self.cfg = cfg
        self.patch_size = patch_size
        self.token_dim = token_dim
        self.image_size = pair(image_size)
        self.hm_size = hm_size
        self.gaze_feature_dim = gaze_feature_dim
        self.encoder_depth = encoder_depth
        self.encoder_num_heads = encoder_num_heads
        self.encoder_num_global_tokens = encoder_num_global_tokens
        self.encoder_mlp_ratio = encoder_mlp_ratio
        self.encoder_use_qkv_bias = encoder_use_qkv_bias
        self.encoder_drop_rate = encoder_drop_rate
        self.encoder_attn_drop_rate = encoder_attn_drop_rate
        self.encoder_drop_path_rate = encoder_drop_path_rate
        self.decoder_feature_dim = decoder_feature_dim
        self.decoder_hooks = decoder_hooks
        self.decoder_hidden_dims = decoder_hidden_dims
        self.decoder_use_bn = decoder_use_bn
        self.vlm_dim = vlm_dim
        self.num_coatt = num_coatt
        window_size = 2*temporal_context+1
        
        # gaze encoder
        self.gaze_encoder = GazeEncoder(
             token_dim = token_dim, 
             feature_dim = gaze_feature_dim
        )
        
        self.gaze_encoder_temporal = TransformerBlock(dim=gaze_feature_dim, num_heads=8, mlp_ratio=0.25, drop_path_rate=0.3)
        
        self.image_tokenizer = SpatialInputTokenizer(                
             num_channels = 3,
             stride_level = 1,
             patch_size = patch_size,
             token_dim = token_dim,
             use_sincos_pos_emb = True,
             is_learnable_pos_emb = False,
             image_size = image_size
        )
        
        ## scene encoder: MultiMAE
        self.encoder = ViTEncoder(
            num_global_tokens = encoder_num_global_tokens,
            token_dim = token_dim,
            depth = encoder_depth,
            num_heads = encoder_num_heads,
            mlp_ratio = encoder_mlp_ratio,
            use_qkv_bias = encoder_use_qkv_bias,
            drop_rate = encoder_drop_rate,
            attn_drop_rate = encoder_attn_drop_rate,
            drop_path_rate = encoder_drop_path_rate
        )  
        ## scene encoder: DinoV2
        # self.encoder = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        
        # scene, person interaction
        self.interaction_indexes=[[0, 2], [3, 5], [6, 8], [9, 11]]
        self.vit_adaptor =  nn.Sequential(*[
            InteractionBlock(dim=token_dim, num_heads=encoder_num_heads, drop_path=0.3, cffn_ratio=0.25)
            for i in range(len(self.interaction_indexes))
        ])
        
        # people interaction
        self.people_interaction = nn.Sequential(*[
            TransformerBlock(dim=token_dim, num_heads=encoder_num_heads, mlp_ratio=0.25, drop_path_rate=0.3)
            for i in range(len(self.interaction_indexes))
        ])
        
#         # people temporal
        self.people_temporal = nn.Sequential(*[
            TransformerBlock(dim=token_dim, num_heads=encoder_num_heads, mlp_ratio=0.25, drop_path_rate=0.3)
            for i in range(len(self.interaction_indexes))
        ])
        
        # temporal position projection layer
        self.temp_emb_new = nn.Parameter(torch.zeros(window_size, gaze_feature_dim))   # final used

        # initialize coatt_emb with N(0, 1)
        self.coatt_dim = proj_feature_dim * len(self.interaction_indexes)
        self.coatt_emb = nn.Parameter(torch.randn(1, 1, self.num_coatt, self.coatt_dim))  # (1, 1, num_coatt, D)

        # coatt-coatt layers
        if self.cfg.model.coatt_coatt_layer=='self':
            self.coatt_coatt_layers = 2
            self.coatt_coatt_interaction = nn.Sequential(*[
                TransformerBlock(dim=self.coatt_dim, num_heads=8, mlp_ratio=0.25)
                for i in range(self.coatt_coatt_layers)
            ])
        else:
            self.coatt_coatt_interaction = nn.Identity()

        # coatt-person layers
        if self.cfg.model.coatt_person_layer=='sep':
            self.people_coatt_layers = 2
            self.people_coatt_interaction = nn.Sequential(*[
                CoattTransformerDecoderBlock(dim=self.coatt_dim, num_heads=8, mlp_ratio=0.25)
                for i in range(self.people_coatt_layers)
            ])
        elif self.cfg.model.coatt_person_layer=='mix':
            self.people_coatt_interaction = nn.Sequential(*[
                TransformerBlock(dim=self.coatt_dim, num_heads=8, mlp_ratio=0.25)
                for i in range(len(self.interaction_indexes))
            ])
        
        self.coatt_token_ffn = nn.Sequential(
            nn.LayerNorm(self.coatt_dim),
            nn.Linear(self.coatt_dim, self.coatt_dim),
            nn.GELU(),
            nn.Linear(self.coatt_dim, self.coatt_dim),
        )

        self.person_tokens_ffn = nn.Sequential(
            nn.LayerNorm(self.coatt_dim),
            nn.Linear(self.coatt_dim, self.coatt_dim),
            nn.GELU(),
            nn.Linear(self.coatt_dim, self.coatt_dim),
        )

        # projection layers for social gaze prediction
        self.gaze_projs = nn.Sequential(*[
            nn.Linear(token_dim, proj_feature_dim, bias=True)
            for i in range(len(self.interaction_indexes))
        ])

        # gaze point, social gaze, inout decoders
        self.output = output
        if output=='heatmap':
            self.gaze_hm_decoder_new = ConditionalDPTDecoder(token_dim = token_dim,
                                                        feature_dim = decoder_feature_dim,
                                                        patch_size = patch_size,
                                                        hooks = decoder_hooks,
                                                        hidden_dims = decoder_hidden_dims,
                                                        use_bn = decoder_use_bn
                                                        )
        else:
            self.gaze_decoder = LinearDecoder(token_dim // 2)
        self.inout_decoder = InOutDecoder(proj_feature_dim*len(self.interaction_indexes))
        
        # social gaze decoders
        self.decoder_lah = LinearDecoderSocialGraph(proj_feature_dim*len(self.interaction_indexes))    # decoder for looking at heads
        self.decoder_sa = LinearDecoderSocialGraph(proj_feature_dim*len(self.interaction_indexes))    # decoder for shared attention
        
    def forward(self, x):
        # Expected x = {"image": image, "heads": heads, "head_bboxes": head_bboxes, "coatt_ids": coatt_ids}
        
        b, t, n, c, h, w = x["heads"].shape # n = total nb of people, t = temporal window

        # Encode Gaze Tokens ===================================================
        gaze_emb = self.gaze_encoder.forward_backbone(x['heads'].view(b*t, n, c, h, w)) 
        gaze_emb = gaze_emb.view(b, t, n, -1)  # (b, t, n, 512)
        
        # Apply temporal attention
        if t > 1:
            # add temporal position embedding
            temp_emb_new = self.temp_emb_new.unsqueeze(0).tile(b, 1, 1).unsqueeze(2)
            gaze_emb = gaze_emb + temp_emb_new
            # perform self-attention
            gaze_emb = self.gaze_encoder_temporal(gaze_emb.permute([0,2,1,3]).reshape(b*n, t, -1), print_att=False)
            gaze_emb = gaze_emb.view(b, n, t, -1).permute([0,2,1,3])
        
        # Predict gaze vector
        gaze_tokens, gaze_vec = self.gaze_encoder.forward_head(gaze_emb.reshape(b*t, n, -1), x['head_bboxes'].view(b*t, n, -1))  # (b*t, n, 768), (b*t, n, 2)
        gaze_vec = gaze_vec.view(b, t, n, -1)
        
        # Tokenize Inputs ===================================================
        b, t, c, h_img, w_img = x["image"].shape
        ## for MultiMAE
        image_tokens, N_H, N_W = self.image_tokenizer(x["image"].view(b*t, c, h_img, w_img)) # (b*t, nt, d) / nt = num_tokens, d = token_dim; (N_H, N_W): number of patches (height, width)
        ## for DinoV2
        # image_tokens = self.encoder.prepare_tokens_with_masks(x['image'].view(b*t, c, h_img, w_img))

        person_tokens = gaze_tokens.view(b*t, n, -1).clone()

        # Apply ViT Adaptor =================================================
        img_layers = []; gaze_layers = []
        for i, layer in enumerate(self.vit_adaptor):
            indexes = self.interaction_indexes[i]
            image_tokens, person_tokens = layer(image_tokens, person_tokens, self.encoder.blocks[indexes[0]:indexes[-1] + 1], x['num_valid_people'])
            # perform people attention
            person_tokens = self.people_interaction[i](person_tokens)
            # Apply temporal attention
            if t > 1:
                person_tokens = self.people_temporal[i](person_tokens.view(b, t, n, -1).permute([0,2,1,3]).reshape(b*n, t, -1))
                person_tokens = person_tokens.view(b, n, t, -1).permute([0,2,1,3]).reshape(b*t, n, -1)            
            # save intermediate outputs
            # img_layers.append(image_tokens[:, 1:])  # for DinoV2, remove class token
            img_layers.append(image_tokens)  # for MultiMAE
            gaze_layers.append(person_tokens)

        # Predict Gaze Heatmap =====================================================
        if self.output=='heatmap':
            # conditional DPT
            gaze_hm = self.gaze_hm_decoder_new(img_layers, gaze_layers, (h_img, w_img))
            _, _, hm_height, hm_width = gaze_hm.shape
            gaze_hm = gaze_hm.view(b, t, n, hm_height, hm_width)

        # project and concat person tokens from each gaze layer
        person_tokens = [self.gaze_projs[i](gaze_layer) for i, gaze_layer in enumerate(gaze_layers)]
        person_tokens = torch.cat(person_tokens, axis=-1)

        # Classify inout ====================================================
        inout = self.inout_decoder(person_tokens.view(b * t * n, -1)) # (b*t*n, 1)

        # compute interaction with co-attention tokens and people tokens
        coatt_tokens = self.coatt_emb
        coatt_tokens = coatt_tokens.view(1, 1, self.num_coatt, self.coatt_dim)
        coatt_tokens = coatt_tokens.expand(b, t, self.num_coatt, self.coatt_dim)  # (b, t, coatt_num, D)
        coatt_tokens = coatt_tokens.view(b*t, self.num_coatt, self.coatt_dim)  # (b*t, coatt_num, D)

        # estimate coatt level by multiplying coatt_tokens_up and person_tokens_up
        person_tokens_coatt = person_tokens.view(b*t, n, self.coatt_dim)  # (b*t, n, D)

        # coatt-coatt interaction
        for i in range(self.coatt_coatt_layers):
            coatt_tokens = self.coatt_coatt_interaction[i](coatt_tokens)

        # coatt as queries, people as keys and values
        if self.cfg.model.coatt_person_layer=='sep':
            for i in range(self.people_coatt_layers):
                coatt_tokens = self.people_coatt_interaction[i](coatt_tokens, person_tokens_coatt)
        elif self.cfg.model.coatt_person_layer=='mix':
            coatt_person_tokens = torch.cat([coatt_tokens, person_tokens_coatt], dim=1)  # (b*t, coatt_num+n, D)
            for i in range(self.people_coatt_layers):
                coatt_person_tokens = self.people_coatt_interaction[i](coatt_person_tokens)
            coatt_tokens = coatt_person_tokens[:, :self.num_coatt, :]  # (b*t, coatt_num, D)
            person_tokens_coatt = coatt_person_tokens[:, self.num_coatt:, :]  # (b*t, n, D)
        
        coatt_tokens_emb = self.coatt_token_ffn(coatt_tokens)
        person_tokens_coatt_emb = self.person_tokens_ffn(person_tokens_coatt)

        coatt_level = torch.einsum('btd,bnd->btn', coatt_tokens_emb, person_tokens_coatt_emb)
        coatt_level_w_prob = torch.sigmoid(coatt_level)

        # Predict Coatt Heatmap by multiplying coatt_level and gaze_hm
        if self.output=='heatmap':
            gaze_hm_view = gaze_hm.view(b, t, 1, n, hm_height, hm_width)
            inout_view = inout.view(b, t, 1, n, 1, 1)
            coatt_level_w_prob_view = coatt_level_w_prob.view(b, t, self.num_coatt, n, 1, 1)
            # coatt_hm = gaze_hm_view * coatt_level_w_prob_view  # (b, t, coatt_num, n, hm_height, hm_width)
            # coatt_hm = (inout_view * gaze_hm_view) * coatt_level_w_prob_view  # (b, t, coatt_num, n, hm_height, hm_width)
            coatt_hm = coatt_level_w_prob_view * gaze_hm_view  # (b, t, coatt_num, n, hm_height, hm_width)
            coatt_hm = coatt_hm.mean(dim=3)  # average over people dimension

        # make person pairs
        indices = torch.tensor(list(itertools.permutations(torch.arange(n), 2))).T # (2, num_pairs)
        num_pairs = indices.shape[1]

        opt_1 = torch.cat([person_tokens[:, [i], :] for i in indices[0]], dim=1) # (b*t, num_pairs, D) - left terms
        opt_2 = torch.cat([person_tokens[:, [j], :] for j in indices[1]], dim=1) # (b*t, num_pairs, D) - right terms
        person_token_pairs = torch.cat([opt_1, opt_2], dim=2) # (b*t, num_pairs, 2*D)
        person_token_pairs = person_token_pairs.reshape(b * t* num_pairs, -1) # (b*t*num_pairs, 2*D)
        
        # Predict social gaze
        lah = self.decoder_lah(person_token_pairs).view(b*t, num_pairs)  # (b*t, num_pairs)
        coatt = self.decoder_sa(person_token_pairs)  # (b*t*num_pairs, 1)
        # laeo = self.decoder_laeo(person_token_pairs)  # (b*t*num_pairs, 1)
        # perform harmonic mean of LAH scores to infer LAEO
        laeo = torch.zeros_like(lah)
        indices = indices.T
        for pi, pair in enumerate(indices):
            corr_idx = torch.where((indices==pair[[1,0]]).prod(-1))[0].item()
            laeo[:, pi] = torch.min(lah[:, pi],lah[:, corr_idx])

        if self.output=='heatmap':
            # return _, gaze_vec, gaze_hm, inout.view(b, t, n), lah.view(b, t, num_pairs), laeo.view(b, t, num_pairs), coatt.view(b, t, num_pairs), coatt_hm
            out = {}
            out['gaze_vec'] = gaze_vec
            out['gaze_hm'] = gaze_hm
            out['inout'] = inout.view(b, t, n)
            out['lah'] = lah.view(b, t, num_pairs)
            out['laeo'] = laeo.view(b, t, num_pairs)
            out['coatt'] = coatt.view(b, t, num_pairs)
            out['coatt_hm'] = coatt_hm
            out['coatt_level'] = coatt_level.view(b, t, self.num_coatt, n)
            out['person_tokens'] = person_tokens.view(b, t, n, -1)

            return out
        else:
            assert False, 'Incorrect output type selected!!!'
# ==================================================================================================================== #
#                                                   InteractNet                                                   #
# ==================================================================================================================== #

# class InteractNet(nn.Module):
#     def __init__(self,
#                  patch_size: int = 16,
#                  token_dim: int = 768,
#                  image_size: int = 224,
#                  gaze_feature_dim: int = 512,
#                  encoder_depth: int = 12,
#                  encoder_num_heads: int = 12,
#                  encoder_num_global_tokens: int = 1,
#                  encoder_mlp_ratio: float = 4.0,
#                  encoder_use_qkv_bias: bool = True,
#                  encoder_drop_rate: float = 0.0,
#                  encoder_attn_drop_rate: float = 0.0,
#                  encoder_drop_path_rate: float = 0.0,
#                  decoder_feature_dim: int = 256,
#                  decoder_hooks: list = [2, 5, 8, 11],
#                  decoder_hidden_dims: list = [96, 192, 384, 768],
#                  decoder_use_bn: bool = False,
#                  proj_feature_dim: int = 128,
#                  vlm_dim: int = 24,
#                  temporal_context: int=2,
#                  hm_size = (64,64),    # width, height
#                  output='heatmap'):
        
#         super().__init__()
        
        
#         self.patch_size = patch_size
#         self.token_dim = token_dim
#         self.image_size = pair(image_size)
#         self.hm_size = hm_size
#         self.gaze_feature_dim = gaze_feature_dim
#         self.encoder_depth = encoder_depth
#         self.encoder_num_heads = encoder_num_heads
#         self.encoder_num_global_tokens = encoder_num_global_tokens
#         self.encoder_mlp_ratio = encoder_mlp_ratio
#         self.encoder_use_qkv_bias = encoder_use_qkv_bias
#         self.encoder_drop_rate = encoder_drop_rate
#         self.encoder_attn_drop_rate = encoder_attn_drop_rate
#         self.encoder_drop_path_rate = encoder_drop_path_rate
#         self.decoder_feature_dim = decoder_feature_dim
#         self.decoder_hooks = decoder_hooks
#         self.decoder_hidden_dims = decoder_hidden_dims
#         self.decoder_use_bn = decoder_use_bn
#         self.vlm_dim = vlm_dim
#         window_size = 2*temporal_context+1
        
#         # gaze encoder
#         self.gaze_encoder = GazeEncoder(
#              token_dim = token_dim, 
#              feature_dim = gaze_feature_dim
#         )
        
#         self.gaze_encoder_temporal = TransformerBlock(dim=gaze_feature_dim, num_heads=8, mlp_ratio=0.25, drop_path_rate=0.3)
        
#         self.image_tokenizer = SpatialInputTokenizer(                
#              num_channels = 3,
#              stride_level = 1,
#              patch_size = patch_size,
#              token_dim = token_dim,
#              use_sincos_pos_emb = True,
#              is_learnable_pos_emb = False,
#              image_size = image_size
#         )
        
#         ## scene encoder: MultiMAE
#         self.encoder = ViTEncoder(
#             num_global_tokens = encoder_num_global_tokens,
#             token_dim = token_dim,
#             depth = encoder_depth,
#             num_heads = encoder_num_heads,
#             mlp_ratio = encoder_mlp_ratio,
#             use_qkv_bias = encoder_use_qkv_bias,
#             drop_rate = encoder_drop_rate,
#             attn_drop_rate = encoder_attn_drop_rate,
#             drop_path_rate = encoder_drop_path_rate
#         )  
#         ## scene encoder: DinoV2
#         # self.encoder = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        
#         # scene, person interaction
#         self.interaction_indexes=[[0, 2], [3, 5], [6, 8], [9, 11]]
#         self.vit_adaptor =  nn.Sequential(*[
#             InteractionBlock(dim=token_dim, num_heads=encoder_num_heads, drop_path=0.3, cffn_ratio=0.25)
#             for i in range(len(self.interaction_indexes))
#         ])
        
#         # people interaction
#         self.people_interaction = nn.Sequential(*[
#             TransformerBlock(dim=token_dim, num_heads=encoder_num_heads, mlp_ratio=0.25, drop_path_rate=0.3)
#             for i in range(len(self.interaction_indexes))
#         ])
        
# #         # people temporal
#         self.people_temporal = nn.Sequential(*[
#             TransformerBlock(dim=token_dim, num_heads=encoder_num_heads, mlp_ratio=0.25, drop_path_rate=0.3)
#             for i in range(len(self.interaction_indexes))
#         ])
        
#         # temporal position projection layer
# ###         self.temp_proj = nn.Linear(1, gaze_feature_dim)
# #         self.temp_emb = nn.Parameter(build_2d_sincos_posemb(1,5,embed_dim=gaze_feature_dim).squeeze().T)
#         self.temp_emb_new = nn.Parameter(torch.zeros(window_size, gaze_feature_dim))   # final used

#         # joint attention layer
#         coatt_num = 10
#         self.coatt_emb = nn.Parameter(torch.zeros(1, coatt_num, token_dim))
#         self.people_coatt_interaction = nn.Sequential(*[
#             InteractionBlock(dim=token_dim, num_heads=encoder_num_heads, mlp_ratio=0.25, drop_path_rate=0.3)
#             for i in range(len(self.interaction_indexes))
#         ])

# #         # speaking status projection layer
#         self.speaking_proj = nn.Linear(1, token_dim)
# #         self.speaking_proj_feat = nn.Linear(512, gaze_feature_dim)
        
# #         # vlm_context projection layer
# #         self.vlm_proj = nn.Linear(self.vlm_dim, token_dim)
        
#         # projection layers for social gaze prediction
#         self.gaze_projs = nn.Sequential(*[
#             nn.Linear(token_dim, proj_feature_dim, bias=True)
#             for i in range(len(self.interaction_indexes))
#         ])
        
#         # gaze point, social gaze, inout decoders
#         self.output = output
#         if output=='heatmap':
# #             self.gaze_hm_decoder = HMDecoder(stride_level = 1,
# #                                              patch_size = patch_size,
# #                                              token_dim = token_dim,
# #                                              image_size = image_size,
# #                                              hm_size = self.hm_size)
# #             self.gaze_hm_decoder = LinearHeatmapDecoder(input_dim = token_dim,
# #                                                          output_hm_size = self.hm_size)
# #             self.gaze_hm_decoder = SimplerHeatmapDecoder(token_dim = token_dim,
# #                                                          h = self.image_size[1]//patch_size,
# #                                                          w = self.image_size[0]//patch_size,
# #                                                          output_hm_size = self.hm_size)
#             self.gaze_hm_decoder_new = ConditionalDPTDecoder(token_dim = token_dim,
#                                                         feature_dim = decoder_feature_dim,
#                                                         patch_size = patch_size,
#                                                         hooks = decoder_hooks,
#                                                         hidden_dims = decoder_hidden_dims,
#                                                         use_bn = decoder_use_bn
#                                                         )
#         else:
#             self.gaze_decoder = LinearDecoder(token_dim // 2)
#         self.inout_decoder = InOutDecoder(proj_feature_dim*len(self.interaction_indexes))
        
#         # speaking status decoder
# #         self.decoder_spk = InOutDecoder(token_dim)
        
#         # social gaze decoders
#         self.decoder_lah = LinearDecoderSocialGraph(proj_feature_dim*len(self.interaction_indexes))    # decoder for looking at heads
# #         self.decoder_laeo = LinearDecoderSocialGraph(proj_feature_dim*len(self.interaction_indexes))    # decoder for laeo
#         self.decoder_sa = LinearDecoderSocialGraph(proj_feature_dim*len(self.interaction_indexes))    # decoder for shared attention
        
        
#     def forward(self, x):
#         # Expected x = {"image": image, "heads": heads, "head_bboxes": head_bboxes, "coatt_ids": coatt_ids}
        
#         b, t, n, c, h, w = x["heads"].shape # n = total nb of people, t = temporal window

#         # Encode Gaze Tokens ===================================================
#         gaze_emb = self.gaze_encoder.forward_backbone(x['heads'].view(b*t, n, c, h, w)) 
#         gaze_emb = gaze_emb.view(b, t, n, -1)  # (b, t, n, 512)
        
# #         # ====== add speaking features ======
# #         speaking_emb = self.speaking_proj_feat(x["speaking_features"].view(-1, 512))    # get speaking emb
# #         speaking_emb = speaking_emb.view(b, t, n, -1)
# #         gaze_emb = gaze_emb + speaking_emb
        
#         # Apply temporal attention
#         if t > 1:
#             # add temporal position embedding
# ###             temp_emb = self.temp_proj(torch.arange(t, dtype=torch.float).unsqueeze(-1).to(gaze_emb.device))
#             temp_emb_new = self.temp_emb_new.unsqueeze(0).tile(b, 1, 1).unsqueeze(2)
#             gaze_emb = gaze_emb + temp_emb_new
#             # perform self-attention
#             gaze_emb = self.gaze_encoder_temporal(gaze_emb.permute([0,2,1,3]).reshape(b*n, t, -1), print_att=False)
#             gaze_emb = gaze_emb.view(b, n, t, -1).permute([0,2,1,3])
        
#         # Predict gaze vector
#         gaze_tokens, gaze_vec = self.gaze_encoder.forward_head(gaze_emb.reshape(b*t, n, -1), x['head_bboxes'].view(b*t, n, -1))  # (b*t, n, 768), (b*t, n, 2)
#         gaze_vec = gaze_vec.view(b, t, n, -1)
        
#         # Tokenize Inputs ===================================================
#         b, t, c, h_img, w_img = x["image"].shape
#         ## for MultiMAE
#         image_tokens, N_H, N_W = self.image_tokenizer(x["image"].view(b*t, c, h_img, w_img)) # (b*t, nt, d) / nt = num_tokens, d = token_dim; (N_H, N_W): number of patches (height, width)
#         ## for DinoV2
#         # image_tokens = self.encoder.prepare_tokens_with_masks(x['image'].view(b*t, c, h_img, w_img))

#         person_tokens = gaze_tokens.view(b*t, n, -1).clone()

# #         # ====== add speaking status embedding ======
# # #         x["speaking"] = x["speaking"]*0-1
# #         speaking_emb = self.speaking_proj(x["speaking"].view(-1, 1))    # get speaking status emb
# #         speaking_emb = speaking_emb.view(b*t, n, -1)
# #         person_tokens = person_tokens + speaking_emb
#         # ====== add vlm embedding ======
# #         vlm_emb = self.vlm_proj(x["person_vlm_context"].view(-1, self.vlm_dim))    # get vlm context emb
# #         vlm_emb = vlm_emb.view(b*t, n, -1)
# #         person_tokens = person_tokens + vlm_emb
        
# #         # Predict speaking status
# #         speaking = self.decoder_spk(person_tokens.view(b*t*n, -1))
# #         speaking = speaking.view(b, t, n)
        
#         # Apply ViT Adaptor =================================================
#         img_layers = []; gaze_layers = []
#         for i, layer in enumerate(self.vit_adaptor):
#             indexes = self.interaction_indexes[i]
#             image_tokens, person_tokens = layer(image_tokens, person_tokens, self.encoder.blocks[indexes[0]:indexes[-1] + 1], x['num_valid_people'])
#             # perform people attention
#             person_tokens = self.people_interaction[i](person_tokens)
# #             # Apply temporal attention
#             if t > 1:
#                 person_tokens = self.people_temporal[i](person_tokens.view(b, t, n, -1).permute([0,2,1,3]).reshape(b*n, t, -1))
#                 person_tokens = person_tokens.view(b, n, t, -1).permute([0,2,1,3]).reshape(b*t, n, -1)            
#             # save intermediate outputs
#             # img_layers.append(image_tokens[:, 1:])  # for DinoV2, remove class token
#             img_layers.append(image_tokens)  # for MultiMAE
#             gaze_layers.append(person_tokens)

#         # Predict Gaze Heatmap =====================================================
#         if self.output=='heatmap':
#             # baseline decoders
# #             hm_width, hm_height = self.hm_size
# #             gaze_hm = self.gaze_hm_decoder(person_tokens.view(b * t * n, -1)) # (b*t*n, d) >> (b*t*n, 64, 64)
# #             gaze_hm = gaze_hm.view(b, t, n, hm_height, hm_width)
# #             # simpler HM decoder
# #             gaze_hm = self.gaze_hm_decoder(img_layers[-1], gaze_layers[-1])
# #             _, _, hm_height, hm_width = gaze_hm.shape
# #             gaze_hm = gaze_hm.view(b, t, n, hm_height, hm_width)
#             # conditional DPT
#             gaze_hm = self.gaze_hm_decoder_new(img_layers, gaze_layers, (h_img, w_img))
#             _, _, hm_height, hm_width = gaze_hm.shape
#             gaze_hm = gaze_hm.view(b, t, n, hm_height, hm_width)

#         # project and concat person tokens from each gaze layer
#         person_tokens = [self.gaze_projs[i](gaze_layer) for i, gaze_layer in enumerate(gaze_layers)]
#         person_tokens = torch.cat(person_tokens, axis=-1)
        
#         # Classify inout ====================================================
#         inout = self.inout_decoder(person_tokens.view(b * t * n, -1)) # (b*t*n, 1)
        
#         # make person pairs
#         indices = torch.tensor(list(itertools.permutations(torch.arange(n), 2))).T # (2, num_pairs)
#         num_pairs = indices.shape[1]

#         opt_1 = torch.cat([person_tokens[:, [i], :] for i in indices[0]], dim=1) # (b*t, num_pairs, D) - left terms
#         opt_2 = torch.cat([person_tokens[:, [j], :] for j in indices[1]], dim=1) # (b*t, num_pairs, D) - right terms
#         person_token_pairs = torch.cat([opt_1, opt_2], dim=2) # (b*t, num_pairs, 2*D)
#         person_token_pairs = person_token_pairs.reshape(b * t* num_pairs, -1) # (b*t*num_pairs, 2*D)
        
#         # Predict social gaze
#         lah = self.decoder_lah(person_token_pairs).view(b*t, num_pairs)  # (b*t, num_pairs)
#         coatt = self.decoder_sa(person_token_pairs)  # (b*t*num_pairs, 1)
# #         laeo = self.decoder_laeo(person_token_pairs)  # (b*t*num_pairs, 1)
#         # perform harmonic mean of LAH scores to infer LAEO
#         laeo = torch.zeros_like(lah)
#         indices = indices.T
#         for pi, pair in enumerate(indices):
#             corr_idx = torch.where((indices==pair[[1,0]]).prod(-1))[0].item()
#             laeo[:, pi] = torch.min(lah[:, pi],lah[:, corr_idx])

#         if self.output=='heatmap':
#             return _, gaze_vec, gaze_hm, inout.view(b, t, n), lah.view(b, t, num_pairs), laeo.view(b, t, num_pairs), coatt.view(b, t, num_pairs)
#         else:
#             print('---------------------------------')
#             print('Incorrect output type selected!!!')
#             print('---------------------------------')
    
    

# ==================================================================================================================== #
#                                                   SHARINGAN BLOCKS                                                   #
# ==================================================================================================================== #

# ****************************************************** #
#                      GAZE ENCODER                      #
# ****************************************************** #
class GazeEncoder(nn.Module):

    def __init__(self, token_dim=768, feature_dim=512):
        super().__init__()

        self.feature_dim = feature_dim
        self.token_dim = token_dim
        
        base = models.resnet18(weights=None)  # type: ignore
        self.backbone = nn.Sequential(*list(base.children())[:-1])
        
        dummy_head = torch.empty((1, 3, 224, 224))
        dummy_head = self.backbone(dummy_head)
        embed_dim = dummy_head.size(1)
        
        self.gaze_proj = nn.Sequential(
            nn.Linear(embed_dim, token_dim),
            nn.ReLU(inplace=True),
            nn.Linear(token_dim, token_dim)
        )
        self.pos_proj = nn.Linear(4, token_dim)
        
        self.gaze_predictor = nn.Sequential(   # self.gaze_predictor
            nn.Linear(embed_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, 2),  # 2 = number of outputs (x, y) unit vector
            nn.Tanh(),
        )
        
        # Initialize weights
        #self.apply(self._init_weights)
        
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0) 


    def forward_backbone(self, head):
        b, n, c, h, w = head.shape

        gaze_emb = self.backbone(head.view(-1, c, h, w)).flatten(1, -1)  # (b*n, embed_dim)

        return gaze_emb
    
    def forward_head(self, gaze_emb, head_bbox):
        b, n, p = head_bbox.shape
        head_bbox_emb = self.pos_proj(head_bbox.view(-1, p))  # (b*n, token_dim)
        
        gaze_token = self.gaze_proj(gaze_emb.view(b*n, -1)) + head_bbox_emb  # (b*n, token_dim)
        gaze_token = gaze_token.view(b, n, -1)  # (b, n, token_dim)

        gaze_vec = self.gaze_predictor(gaze_emb.view(b*n, -1))  # (b*n, 2)
        gaze_vec = F.normalize(gaze_vec, p=2, dim=1)  # normalize to unit vector
        gaze_vec = gaze_vec.view(b, n, -1)  # (b, n, 2)
        
        return gaze_token, gaze_vec


# ****************************************************** #
#                     LINEAR DECODER                     #
# ****************************************************** #

class ResidualLinearBlock(nn.Module):
    def __init__(self, feature_dim, scale=2):
        super().__init__()
        self.scale = scale
        self.feature_dim = feature_dim
        
        self.fc1 = nn.Linear(feature_dim, feature_dim // scale, bias=False)
        self.bn1 = nn.BatchNorm1d(feature_dim // scale)
        self.fc2 = nn.Linear(feature_dim // scale, feature_dim // scale**2, bias=False)
        self.bn2 = nn.BatchNorm1d(feature_dim // scale**2)
        self.res_fc = nn.Linear(feature_dim, feature_dim // scale**2)
        
    def forward(self, x):
        z = torch.relu(self.bn1(self.fc1(x)))
        o = torch.relu(self.bn2(self.fc2(z)) + self.res_fc(x))
        return o
    

class LinearDecoder(nn.Module):
    def __init__(self, token_dim):
        super().__init__()
        self.token_dim = token_dim
        
        self.block1 = ResidualLinearBlock(2 * token_dim, scale=2)
        self.block2 = ResidualLinearBlock(token_dim // 2, scale=2)
        self.fc = nn.Linear(token_dim // 8, 2)
        
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.fc(x)
        x = torch.sigmoid(x)
        return x
        
    
class InOutDecoder(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        
        self.block1 = ResidualLinearBlock(dim)
        self.block2 = ResidualLinearBlock(dim // 4)
        self.fc = nn.Linear(dim // 16, 1)
        
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.fc(x)
        return x  
    
    
class LinearDecoderSocialGraph(nn.Module):
    def __init__(self, token_dim):
        super().__init__()
        self.token_dim = token_dim
        
        scale = 4
        self.block1 = ResidualLinearBlock(2*token_dim, scale=scale)
#         self.block2 = ResidualLinearBlock(2*token_dim // scale**2, scale=scale)    # removed for some experiments
        self.fc = nn.Linear(token_dim*2 // scale**2, 1)
        
    def forward(self, edge_tokens):
        
        x = self.block1(edge_tokens)
#         x = self.block2(x)
        # readout layer
        x = self.fc(x)
        return x
    

    
    
# ****************************************************** #
#                      HEATMAP DECODERS                  #
# ****************************************************** #
class ConditionalDPTDecoder(nn.Module):
    '''
    Adapted re-assemble stage of standard DPT for person-conditioned gaze heatmap prediction
    '''
    def __init__(self,
                 patch_size: Union[int, Tuple[int, int]] = 16,
                 hooks: List[int] = [2, 5, 8, 11],
                 hidden_dims: List[int] = [96, 192, 384, 768],
                 token_dim: int = 768,
                 feature_dim: int = 128,
                 use_bn: bool = True):
        super().__init__()
        
        self.patch_size = pair(patch_size)
        self.hooks = hooks
        self.token_dim = token_dim
        self.hidden_dims = hidden_dims
        self.feature_dim = feature_dim
        self.use_bn = use_bn

        self.patch_h = self.patch_size[0]
        self.patch_w = self.patch_size[1]
        
        assert len(hooks) <= 4, f"The argument hooks can't have more than 4 elements."
        self.factors = [4, 8, 16, 32][-len(hooks):]
        self.reassemble_blocks = nn.ModuleDict({
            f"r{factor}": Reassemble(factor, hidden_dims[idx], feature_dim=feature_dim, token_dim=token_dim) \
            for idx, factor in enumerate(self.factors)
        })
        
        self.fusion_blocks = nn.ModuleDict({
            f"f{factor}": FusionBlock(feature_dim, use_bn=use_bn) \
            for idx, factor in enumerate(self.factors)
        })
        
        self.gaze_projs = nn.ModuleDict({
            f"g{factor}": nn.Linear(token_dim, feature_dim, bias=True) \
            for idx, factor in enumerate(self.factors)
        })
        
        self.head = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim // 2, kernel_size=3, stride=1, padding=1),
            #Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
            nn.ReLU(True),
            nn.Conv2d(feature_dim // 2, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0)
        )
        
    def forward(self, img_layers, gaze_layers, img_size):
        img_h, img_w = img_size
        feat_h = img_h // self.patch_h
        feat_w = img_w // self.patch_w

        b, n, _ = gaze_layers[-1].shape

        # Reshape tokens into spatial representation
        img_layers = [einops.rearrange(l, 'b (fh fw) d -> b d fh fw', fh=feat_h, fw=feat_w) for l in img_layers]
        
        # Apply reassemble and fusion blocks
        for idx, (factor, img_layer, gaze_layer) in enumerate(zip(self.factors[::-1], img_layers[::-1], gaze_layers[::-1])):
            f = self.reassemble_blocks[f"r{factor}"](img_layer)
            _, d, h, w = f.shape
            g = self.gaze_projs[f"g{factor}"](gaze_layer) # (b, n, d) > # (b, n, d')
            f = torch.einsum("bdhw,bnd->bndhw", f, g).view(-1, self.feature_dim, h, w) # (b, n, d', H/32, W/32) > (b*n, d', H/32, W/32)
            #f = f.unsqueeze(1).repeat(1, n, 1, 1, 1) + g.view(b, n, d, 1, 1) # (b, n, d', H/32, W/32)
            #f = f.view(-1, self.feature_dim, h, w) # (b, n, d', H/32, W/32) > (b*n, d', H/32, W/32)
            if idx == 0:
                z = self.fusion_blocks[f"f{factor}"](f) # (b*n, d', H/16, W/16)
            else:
                z = self.fusion_blocks[f"f{factor}"](f, z) # (b*n, d', H/16, W/16)
                
        # Apply prediction head and downscale (224 > 64)
        z = self.head(z) # (b*n, d', H/2, W/2) > (b*n, 1, H/2, W/2)
        z = F.interpolate(z, size=(64, 64), mode="bilinear", align_corners=False) # (b*n, 1, H, W) > (b*n, 1, 64, 64)
        z = z.view(b, n, 64, 64) # (b*n, 1, 64, 64) > (b, n, 64, 64)
        return z
    

    
class LinearHeatmapDecoder(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=1024, output_hm_size=64):
        """
        Project and reshape the output gaze token into a heatmap.
        """
        super().__init__()
        
        self.output_hm_size = output_hm_size
        
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.res_fc = nn.Linear(input_dim, hidden_dim)
        self.out_fc = nn.Linear(hidden_dim, output_hm_size[0]*output_hm_size[1])
        
    def forward(self, x):
        z = torch.relu(self.bn1(self.fc1(x)))
        o = torch.relu(self.bn2(self.fc2(z)) + self.res_fc(x))
        o = self.out_fc(o)
        o = o.view(-1, self.output_hm_size[1], self.output_hm_size[0])
        return o



class SimplerHeatmapDecoder(nn.Module):
    """
    Project image tokens and gaze tokens, then perform a pixel-wise dot-product before 
    upsampling to 64x64 using a resize operation.
    """
    def __init__(self, token_dim=768, h=14, w=14, output_hm_size = (64,64), factor = 6):
        super().__init__()
        
        self.h, self.w = h, w
        self.num_img_tokens = h * w
        self.output_hm_size = output_hm_size
        
        self.img_proj = MLP(token_dim, token_dim, token_dim // factor, drop_rate = 0.)
        self.gaze_proj = MLP(token_dim, token_dim, token_dim // factor, drop_rate = 0.)
        
    def forward(self, x_img, x_gaze):
        
        b, n, d = x_gaze.shape
        
        x_img = self.img_proj(x_img).permute(0, 2, 1) # (b, h*w, d) > (b, h*w, d') > (b, d', h*w)
        x_gaze = self.gaze_proj(x_gaze) # (b, n, d) > (b, n, d')
        
        heatmap = (x_gaze @ x_img).view(b, n, self.h, self.w) # (b, n, h*w) > (b, n, h', w')
        heatmap = TF.resize(heatmap, (self.output_hm_size[1], self.output_hm_size[0]), antialias=True)
        
        return heatmap

    
    
class HMDecoder(nn.Module):
    '''
    Perform an element-wise dot product of a 64x64xtoken_dim positional embedding with the person token then smooth
    '''
    def __init__(self, 
                 patch_size: Union[int, Tuple[int,int]],
                 stride_level: int,
                 token_dim: int = 768,
                 image_size: Union[int, Tuple[int]] = 224,
                 hm_size: Union[int, Tuple[int]] = 64):
        super().__init__()
        patch_size = pair(patch_size)
        image_size = pair(image_size)
        hm_size = pair(hm_size)
        
        # build position embedding
        P_H = max(1, patch_size[0] // stride_level)
        P_W = max(1, patch_size[1] // stride_level)
        h_pos_emb = image_size[0] // (stride_level * P_H)
        w_pos_emb = image_size[1] // (stride_level * P_W)
        self.pos_emb = build_2d_sincos_posemb(h=h_pos_emb, w=w_pos_emb, embed_dim=token_dim)
        self.pos_emb = F.interpolate(self.pos_emb, size=hm_size, mode='bicubic', align_corners=False)
        
        # smoothing conv layers
        self.head = nn.Sequential(
            nn.Conv2d(token_dim, token_dim // 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(token_dim//2),
            nn.ReLU(True),
            nn.Conv2d(token_dim // 2, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0)
        )
        
    def forward(self, x): 
        device = x.device
        self.pos_emb = self.pos_emb.to(device)
        x = self.pos_emb + x.unsqueeze(-1).unsqueeze(-1)
        x = self.head(x)
        return x
    

# ****************************************************** #
#                      VIT ENCODER                       #
# ****************************************************** #
class ViTEncoder(nn.Module):        
    def __init__(self,
                 num_global_tokens: int = 1,
                 token_dim: int = 768,
                 depth: int = 12,
                 num_heads: int = 12,
                 mlp_ratio: float = 4.0,
                 use_qkv_bias: bool = True,
                 drop_rate: float = 0.0,
                 attn_drop_rate: float = 0.0,
                 drop_path_rate: float = 0.0):
        super().__init__()


        # Add global tokens
        self.num_global_tokens = num_global_tokens
        self.global_tokens = nn.Parameter(torch.zeros(1, num_global_tokens, token_dim))
        nn.init.trunc_normal_(self.global_tokens, std=0.02)
        
        # Add encoder layers
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            TransformerBlock(
                dim=token_dim, 
                num_heads=num_heads, 
                mlp_ratio=mlp_ratio, 
                use_qkv_bias=use_qkv_bias,
                drop_rate=drop_rate, 
                attn_drop_rate=attn_drop_rate, 
                drop_path_rate=dpr[i]
            )
            for i in range(depth)
        ])
        
        # Initialize weights
        self.apply(self._init_weights)
        # Initialize the weights of Q, K, V separately
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear) and ('qkv' in name):
                val = math.sqrt(6. / float(m.weight.shape[0] // 3 + m.weight.shape[1]))
                nn.init.uniform_(m.weight, -val, val)
        
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)    


    def __len__(self):
        return len(self.encoder)
     

    def forward(self, input_tokens: torch.Tensor, return_all_layers: bool = True):
        
        # Add global tokens to input tokens
        global_tokens = einops.repeat(self.global_tokens, '() n d -> b n d', b=len(input_tokens))
        input_tokens = torch.cat([input_tokens, global_tokens], dim=1)

        # Pass tokens through Transformer
        if not return_all_layers:
            encoder_tokens = self.blocks(input_tokens)
        else:
            # Optionally access every intermediate layer
            encoder_tokens = []
            tokens = input_tokens
            for block in self.blocks:
                tokens = block(tokens)
                encoder_tokens.append(tokens)
                
        return encoder_tokens
    
    
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., use_qkv_bias=False, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, num_heads=num_heads, use_qkv_bias=use_qkv_bias, attn_drop_rate=attn_drop_rate, proj_drop_rate=drop_rate)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = MLP(in_features=dim, hidden_features=int(dim * mlp_ratio), drop_rate=drop_rate)

    def forward(self, x, print_att=False):
        x = x + self.drop_path(self.attn(self.norm1(x), print_att))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

# coatt transformer decoder block
class CoattTransformerDecoderBlock(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_ratio=4., use_kv_bias=False, use_q_bias=False, attn_drop_rate=0., proj_drop_rate=0., drop_path_rate=0.):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.cross_attn = CoAttCrossAttention(dim, num_heads=num_heads, use_kv_bias=use_kv_bias, use_q_bias=use_q_bias)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = MLP(in_features=dim, hidden_features=int(dim * mlp_ratio), drop_rate=proj_drop_rate)

    # cross attention for coatt tokens to obtain the information from gaze tokens
    def forward(self, coatt_tokens, gaze_tokens):
        # gaze_tokens: (b, n, d), coatt_tokens: (b, m, d)
        # where b = batch size, n = number of people, m = number of coatt tokens, d = token dimension
        coatt_tokens = coatt_tokens + self.drop_path(self.cross_attn(self.norm1(coatt_tokens), gaze_tokens))
        coatt_tokens = coatt_tokens + self.drop_path(self.mlp(self.norm2(coatt_tokens)))
        return coatt_tokens

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop_rate=0.):
        super().__init__()
        
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop_rate)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x 

    
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, use_qkv_bias=False, attn_drop_rate=0., proj_drop_rate=0.):
        super().__init__()
        
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=use_qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_rate)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_rate)

    def forward(self, x, print_att):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        if print_att:
            print(attn.shape)
            print(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    
def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return 'p={}'.format(self.drop_prob)
    
    
# ****************************************************** #
#                SPATIAL INPUT TOKENIZER                 #
# ****************************************************** #
class SpatialInputTokenizer(nn.Module):
    """Tokenizer for spatial inputs, like images or heatmaps.
    Creates tokens from patches over the input image.

    :param num_channels: Number of input channels of the image/feature map
    :param stride_level: Stride level compared to the full-sized image (e.g. 4 for 1/4th the size of the image).
    :param patch_size: Int or tuple of the patch size over the full image size. Patch size for smaller inputs will be computed accordingly.
    :param token_dim: Dimension of output tokens.
    :param use_sincos_pos_emb: Set to True (default) to use fixed 2D sin-cos positional embeddings.
    :param is_learnable_pos_emb: Set to True to learn positional embeddings instead.
    :param image_size: Default image size. Used to initialize size of positional embeddings.
    """
    def __init__(self,                 
                 num_channels: int,
                 stride_level: int,
                 patch_size: Union[int, Tuple[int,int]],
                 token_dim: int = 768,
                 use_sincos_pos_emb: bool = True,
                 is_learnable_pos_emb: bool = False,
                 image_size: Union[int, Tuple[int]] = 224):
        
        super().__init__()
        self.num_channels = num_channels
        self.stride_level = stride_level
        self.patch_size = pair(patch_size)
        self.token_dim = token_dim
        self.use_sincos_pos_emb = use_sincos_pos_emb
        self.is_learnable_pos_emb = is_learnable_pos_emb
        self.image_size = pair(image_size)
        self.num_patches = (self.image_size[0] // self.patch_size[0]) * (self.image_size[1] // self.patch_size[1])

        self.P_H = max(1, self.patch_size[0] // stride_level)
        self.P_W = max(1, self.patch_size[1] // stride_level)
        
        self._init_pos_emb()
        self.proj = nn.Conv2d(
            in_channels=self.num_channels, 
            out_channels=self.token_dim,
            kernel_size=(self.P_H, self.P_W), 
            stride=(self.P_H, self.P_W)
        )
        
    def _init_pos_emb(self):
        # Fixed-size positional embeddings. Can be interpolated to different input sizes
        h_pos_emb = self.image_size[0] // (self.stride_level * self.P_H)
        w_pos_emb = self.image_size[1] // (self.stride_level * self.P_W)
        
        if self.use_sincos_pos_emb:
            self.pos_emb = build_2d_sincos_posemb(h=h_pos_emb, w=w_pos_emb, embed_dim=self.token_dim)
            self.pos_emb = nn.Parameter(self.pos_emb, requires_grad=self.is_learnable_pos_emb)
        else:
            self.pos_emb = nn.Parameter(torch.zeros(1, self.token_dim, h_pos_emb, w_pos_emb))
            nn.init.trunc_normal_(self.pos_emb, mean=0.0, std=0.02, a=-2.0, b=2.0)

        
    def forward(self, x):
        # input.shape = BxCxHxW >> output.shape = BxNxD (where N=n_tokens, D=token_dim)
        
        B, C, H, W = x.shape
        
        assert (H % self.P_H == 0) and (W % self.P_W == 0), f'Image size {H}x{W} must be divisible by patch size {self.P_H}x{self.P_W}'
        N_H, N_W = H // self.P_H, W // self.P_W # Number of patches in height and width

        # Create tokens [B, C, PH, PW] >> [B, D, H, W] >> [B, (H*W), D]
        x_tokens = einops.rearrange(self.proj(x), 'b d h w -> b (h w) d')

        # Create positional embedding
        x_pos_emb = F.interpolate(self.pos_emb, size=(N_H, N_W), mode='bicubic', align_corners=False)
        x_pos_emb = einops.rearrange(x_pos_emb, 'b d h w -> b (h w) d')

        # Add patches and positional embeddings
        x = x_tokens + x_pos_emb

        return x, N_H, N_W
    

# ****************************************************** #
#                      DPT DECODER                       #
# ****************************************************** #
class DPTDecoder(nn.Module):
    def __init__(self,
                 stride_level: int = 1,
                 patch_size: Union[int, Tuple[int, int]] = 16,
                 hooks: List[int] = [2, 5, 8, 11],
                 hidden_dims: List[int] = [96, 192, 384, 768],
                 token_dim: int = 768,
                 feature_dim: int = 256,
                 use_bn: bool = False):
        super().__init__()
        
        self.stride_level = stride_level
        self.patch_size = pair(patch_size)
        self.hooks = hooks
        self.token_dim = token_dim
        self.hidden_dims = hidden_dims
        self.feature_dim = feature_dim
        self.use_bn = use_bn

        self.P_H = max(1, self.patch_size[0] // stride_level)
        self.P_W = max(1, self.patch_size[1] // stride_level)
        
        self.reassemble_blocks = nn.ModuleDict({
            f"r{factor}": Reassemble(factor, hidden_dims[idx], feature_dim=feature_dim, token_dim=token_dim) \
            for idx, factor in enumerate([4, 8, 16, 32])
        })
        
        self.fusion_blocks = nn.ModuleDict({
            f"f{factor}": FusionBlock(feature_dim, use_bn=use_bn) \
            for idx, factor in enumerate([4, 8, 16, 32])
        })
        
        self.head = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim // 2, kernel_size=3, stride=1, padding=1),
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(feature_dim // 2, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0)
        )
        
        # Initialize weights
        #self.apply(self._init_weights)
        
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0) 

    
    def filter_tokens(self, tokens, start, end):
        """Takes a sequence of tokens as input and keeps only the ones to be used by the decoder.
        """
        # tokens.shape = BxTxD (B=batch, T=tokens, D=dim)
        return tokens[:, start:end, :]

        
    def forward(self, x):
        H, W = x["image_size"]
        N_H = H // (self.stride_level * self.P_H)
        N_W = W // (self.stride_level * self.P_W)
        
        # Retrieve intermediate encoder activations
        layers = x["input"]
        layers = [layers[hook] for hook in self.hooks]
        
        # Filter output tokens
        layers = [self.filter_tokens(tokens, 0, N_W * N_H) for tokens in layers]
        
        # Reshape tokens into spatial representation
        layers = [einops.rearrange(l, 'b (nh nw) d -> b d nh nw', nh=N_H, nw=N_W) for l in layers]
        
        # Apply reassemble and fusion blocks
        z32 = self.fusion_blocks.f32(self.reassemble_blocks.r32(layers[3])) # z32 = (B, C, H/16, W/16)
        z16 = self.fusion_blocks.f16(self.reassemble_blocks.r16(layers[2]), z32) # z16 = (B, C, H/8, W/8)
        z8 = self.fusion_blocks.f8(self.reassemble_blocks.r8(layers[1]), z16) # z8 = (B, C, H/4, W/4)
        z4 = self.fusion_blocks.f4(self.reassemble_blocks.r4(layers[0]), z8) # z4 = (B, C, H/2, W/2)
        
        # Apply prediction head
        z = self.head(z4) # z = (B, C, H, W)
        
        return z
    

class Interpolate(nn.Module):
    """Interpolation module."""

    def __init__(self, scale_factor, mode, align_corners=False):
        """Init.
        Args:
            scale_factor (float): scaling
            mode (str): interpolation mode
        """
        super(Interpolate, self).__init__()

        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        """Forward pass.
        Args:
            x (tensor): input
        Returns:
            tensor: interpolated data
        """

        x = self.interp(
            x,
            scale_factor=self.scale_factor,
            mode=self.mode,
            align_corners=self.align_corners,
        )

        return x
    
    def __repr__(self):
        return f"Interpolate(scale_factor={self.scale_factor}, mode={self.mode}, align_corners={self.align_corners})"


class Reassemble(nn.Module):
    def __init__(self, factor, hidden_dim, feature_dim=256, token_dim=768):
        super().__init__()
        
        assert factor in [4, 8, 16, 32], f"Argument `factor` not supported. Choose from [0.5, 4, 8, 16, 32]."
        self.factor = factor
        self.hidden_dim = hidden_dim
        self.feature_dim = feature_dim
        self.token_dim = token_dim
        
        if factor == 4:
            self.resample = nn.Sequential(
                nn.Conv2d(token_dim, hidden_dim, kernel_size=1, stride=1, padding=0, bias=True),
                nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_size=4, stride=4, padding=0, bias=True),
            )
            self.proj = nn.Conv2d(hidden_dim, feature_dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        elif factor == 8:
            self.resample = nn.Sequential(
                nn.Conv2d(token_dim, hidden_dim, kernel_size=1, stride=1, padding=0, bias=True),
                nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_size=2, stride=2, padding=0, bias=True)
            )
            self.proj = nn.Conv2d(hidden_dim, feature_dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        elif factor == 16:
            self.resample = nn.Sequential(
                nn.Conv2d(token_dim, hidden_dim, kernel_size=1, stride=1, padding=0, bias=True),
            )
            self.proj = nn.Conv2d(hidden_dim, feature_dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        elif factor == 32:
            self.resample = nn.Sequential(
                nn.Conv2d(token_dim, hidden_dim, kernel_size=1, stride=1, padding=0, bias=True),
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1, bias=True)
            )
            self.proj = nn.Conv2d(hidden_dim, feature_dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        
    def forward(self, x):
        x = self.resample(x)
        x = self.proj(x)
        return x


class ResidualConvUnit(nn.Module):
    """Residual convolution module."""

    def __init__(self, feature_dim, use_bn=False):
        """Init.
        Args:
            features (int): dimension of feature maps
            use_bn (bool): whether to use batch normalization in the Residual Conv Units.
        """
        super().__init__()

        self.feature_dim = feature_dim
        self.use_bn = use_bn
        
        modules = nn.ModuleList([
            nn.ReLU(inplace=False),
            nn.Conv2d(feature_dim, feature_dim, kernel_size=3, stride=1, padding=1, bias=(not self.use_bn)),
            nn.ReLU(inplace=False),
            nn.Conv2d(feature_dim, feature_dim, kernel_size=3, stride=1, padding=1, bias=(not self.use_bn)),
        ])
        if self.use_bn:
            modules.insert(2, nn.BatchNorm2d(feature_dim))
            modules.insert(5, nn.BatchNorm2d(feature_dim))
        self.residual_module = nn.Sequential(*modules) 

    def forward(self, x):
        z = self.residual_module(x)
        return z + x


class FusionBlock(nn.Module):
    def __init__(self, feature_dim, use_bn=False):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.use_bn = use_bn
        
        self.rcu1 = ResidualConvUnit(feature_dim, use_bn=use_bn)
        self.rcu2 = ResidualConvUnit(feature_dim, use_bn=use_bn)
        self.resample = Interpolate(2, "bilinear", align_corners=True)
        self.proj = nn.Conv2d(feature_dim, feature_dim, kernel_size=1, stride=1, padding=0, bias=True)
        
    def forward(self, *xs):
        assert 1 <= len(xs) <= 2, f"Can only accept inputs of length <= 2. Received len(xs)={len(xs)}"

        z = self.rcu1(xs[0])
        if len(xs) == 2:    
            z = z + xs[1]
        z = self.rcu2(z)
        z = self.resample(z)
        z = self.proj(z)
        
        return z