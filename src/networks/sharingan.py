import math
import sys
from typing import Dict, List, Optional, Tuple, Union

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from src.utils import pair, build_2d_sincos_posemb




# ==================================================================================================================== #
#                                                SHARINGAN ARCHITECTURE                                                #
# ==================================================================================================================== #
class Sharingan(nn.Module):
    def __init__(self,
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
                 decoder_use_bn: bool = False):
        
        super().__init__()
        
        
        self.patch_size = patch_size
        self.token_dim = token_dim
        self.image_size = pair(image_size)
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
        
        self.gaze_encoder = GazeEncoder(
             token_dim = token_dim, 
             feature_dim = gaze_feature_dim
        )
        
        self.image_tokenizer = SpatialInputTokenizer(                
             num_channels = 3,
             stride_level = 1,
             patch_size = patch_size,
             token_dim = token_dim,
             use_sincos_pos_emb = True,
             is_learnable_pos_emb = False,
             image_size = image_size
        )
        
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
        
        self.gaze_decoder = LinearDecoder(token_dim // 2)
        
        self.inout_decoder = InOutDecoder(2 * token_dim)
        
        
    def forward(self, x):
        # Expected x = {"image": image, "heads": heads, "head_bboxes": head_bboxes, "coatt_ids": coatt_ids}
        
        b, n, c, h, w = x["heads"].shape # n = total nb of people
            
        # Encode Gaze Tokens ===================================================
        gaze_tokens, gaze_vec = self.gaze_encoder(x["heads"], x["head_bboxes"]) # (b, n, d), (b, n, 2)
        
        # Tokenize Inputs ===================================================
        image_tokens = self.image_tokenizer(x["image"]) # (b, t, d) / t = num_tokens, d = token_dim
        input_tokens = torch.cat([image_tokens, gaze_tokens], dim=1) # (b, t+n, d)

        # Encode Tokens =====================================================
        output_tokens = self.encoder(input_tokens, return_all_layers=False) # (b, t+n+1, d) / +1 for global token

        # Decode Tokens =====================================================
        output_gaze_tokens = output_tokens[:, -n-1:-1, :].reshape(b * n, -1) # keep only gaze tokens (last position = global token)
        gaze_pt = self.gaze_decoder(output_gaze_tokens) # (b*n, d) >> (b*n, 2)
        gaze_pt = gaze_pt.view(b, n, -1) # (b, n, 2)

        # Classify inout ====================================================
        inout_input_tokens = torch.cat([output_gaze_tokens, gaze_tokens.view(b * n, -1)], dim=1) # (b*n, 2*d)
        inout = self.inout_decoder(inout_input_tokens) # (b*n, 1)
        inout = inout.view(b, n, -1) # (b, n, 1)

        return gaze_vec, gaze_pt, inout

    
    

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
        
        self.gaze_predictor = nn.Sequential(
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


    def forward(self, head, head_bbox):
        b, n, p = head_bbox.shape
        b, n, c, h, w = head.shape

        head_bbox_emb = self.pos_proj(head_bbox.view(-1, p))  # (b*n, token_dim)
        gaze_emb = self.backbone(head.view(-1, c, h, w)).flatten(1, -1)  # (b*n, embed_dim)

        gaze_token = self.gaze_proj(gaze_emb) + head_bbox_emb  # (b*n, token_dim)
        gaze_token = gaze_token.view(b, n, -1)  # (b, n, token_dim)

        gaze_vec = self.gaze_predictor(gaze_emb)  # (b*n, 2)
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
        self.encoder = nn.Sequential(*[
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
            encoder_tokens = self.encoder(input_tokens)
        else:
            # Optionally access every intermediate layer
            encoder_tokens = []
            tokens = input_tokens
            for block in self.encoder:
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

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
    

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

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

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

        return x
    

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