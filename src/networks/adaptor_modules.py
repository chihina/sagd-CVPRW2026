import logging
import itertools
from functools import partial

import torch
import torch.nn as nn
from timm.models.layers import DropPath
import torch.utils.checkpoint as cp

import torch_geometric
import torch_geometric.data as geom_data
import torch_geometric.nn as geom_nn

_logger = logging.getLogger(__name__)

class CoAttCrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, use_kv_bias=False, use_q_bias=False):
        super().__init__()
        
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.kv = nn.Linear(dim, dim * 2, bias=use_kv_bias)
        self.q = nn.Linear(dim, dim, bias=use_q_bias)
        self.proj = nn.Linear(dim, dim)
        
    def forward(self, coatt_tokens, gaze_token):
        B, NP, C = coatt_tokens.shape
        B, N, C = gaze_token.shape
        
        kv = self.kv(gaze_token).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0) # (b, nh, n, dh)
        
        # (b, np, d) >> (b, nh, np, dh) where np=num of people
        q = self.q(coatt_tokens).reshape(B, NP, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3) 

        attn = (q @ k.transpose(-2, -1)) * self.scale # (b, nh, np, n)
        attn = attn.softmax(dim=-1)

        o = (attn @ v).transpose(1, 2).reshape(B, NP, C) # (b, np, d)
        o = self.proj(o)
        return o


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, use_kv_bias=False, use_q_bias=False):
        super().__init__()
        
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.kv = nn.Linear(dim, dim * 2, bias=use_kv_bias)
        self.q = nn.Linear(dim, dim, bias=use_q_bias)
        self.proj = nn.Linear(dim, dim)
        
        
    def forward(self, gaze_token, img_tokens):
        B, N, C = img_tokens.shape
        _, NP, _ = gaze_token.shape
        
        kv = self.kv(img_tokens).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0) # (b, nh, n, dh)
        
        # (b, np, d) >> (b, nh, np, dh) where np=num of people
        q = self.q(gaze_token).reshape(B, NP, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3) 

        attn = (q @ k.transpose(-2, -1)) * self.scale # (b, nh, np, n)
        attn = attn.softmax(dim=-1)

        o = (attn @ v).transpose(1, 2).reshape(B, NP, C) # (b, np, d)
        o = self.proj(o)
        return o


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


class Extractor(nn.Module):
    def __init__(self, dim, num_heads=6, with_cffn=True, cffn_ratio=0.25, drop=0., drop_path=0.,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), with_cp=False):
        super().__init__()
        self.query_norm = norm_layer(dim)
        self.feat_norm = norm_layer(dim)
        self.attn = CrossAttention(dim=dim, num_heads=num_heads)
        self.with_cffn = with_cffn
        self.with_cp = with_cp
        if with_cffn:
            self.ffn = MLP(in_features=dim, hidden_features=int(dim * cffn_ratio), drop_rate=drop)
            self.ffn_norm = norm_layer(dim)
            self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
    
    def forward(self, query, feat):
        
        def _inner_forward(query, feat):
            
            attn = self.attn(self.query_norm(query), self.feat_norm(feat))
            query = query + attn
            
            if self.with_cffn:
                query = query + self.drop_path(self.ffn(self.ffn_norm(query)))
            return query
        
        if self.with_cp and query.requires_grad:
            query = cp.checkpoint(_inner_forward, query, feat)
        else:
            query = _inner_forward(query, feat)
        
        return query


class Injector(nn.Module):
    def __init__(self, dim, num_heads=6, norm_layer=partial(nn.LayerNorm, eps=1e-6), init_values=0., with_cp=False, same_norm=False):
        super().__init__()
        self.with_cp = with_cp
        self.query_norm = norm_layer(dim)
        if same_norm:
            self.feature_norm = self.query_norm
        else:
            self.feat_norm = norm_layer(dim)
        self.attn = CrossAttention(dim=dim, num_heads=num_heads)
        self.gamma = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
    
    def forward(self, query, feat):
                
        def _inner_forward(query, feat):
            
            attn = self.attn(self.query_norm(query), self.feat_norm(feat))
            return query + self.gamma * attn
        
        if self.with_cp and query.requires_grad:
            query = cp.checkpoint(_inner_forward, query, feat)
        else:
            query = _inner_forward(query, feat)
        
        return query

    
class GraphGaze(nn.Module):
    def __init__(self, in_channels=768, hidden_channels=96, heads=8):
        super().__init__()
        self.conv1 = geom_nn.GATv2Conv(in_channels,  hidden_channels, heads=heads, concat=True)
        self.norm1 = partial(nn.LayerNorm, eps=1e-6)(hidden_channels*heads)
#         self.conv2 = geom_nn.GATv2Conv(hidden_channels*heads, hidden_channels, heads=heads, concat=True)
#         self.norm2 = partial(nn.LayerNorm, eps=1e-6)(hidden_channels*heads)
        self.readout = nn.Linear(hidden_channels*heads, 768)

    def forward(self, x, num_valid_people):
        batch_size, num_people, token_dim = x.shape
        edge_index_list = self.build_edge_index(batch_size, num_people, num_valid_people, x.device)
        batch = self.build_graph_batch(x, edge_index_list)
        x = self.conv1(batch.x, batch.edge_index.int())
        x = nn.GELU()(self.norm1(x))
#         x = self.conv2(x, batch.edge_index.int())
#         x = nn.GELU()(self.norm2(x))
        x = self.readout(x)
        
        return x.view(batch_size, num_people, -1)

    def build_edge_index(self, batch_size, num_people, num_valid_people, device):
        edge_index_list = [
                torch.tensor(list(itertools.permutations(torch.arange(num_people-num_valid_people[i], num_people, device=device), 2)), device=device).T
                for i in range(batch_size)
            ]
        return edge_index_list

    def build_graph_batch(self, x,  edge_index_list):
        data_list = [geom_data.Data(x=x_i, edge_index=ei) for (x_i, ei) in zip(x, edge_index_list)] 
        batch = geom_data.Batch.from_data_list(data_list)
        return batch
    

class InteractionBlock(nn.Module):
    def __init__(self, dim, num_heads=6, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 drop=0., drop_path=0., with_cffn=True, cffn_ratio=0.25, init_values=0.,
                 extra_extractor=False, with_cp=False):
        super().__init__()
        
        self.injector = Injector(dim=dim, num_heads=num_heads, init_values=init_values,
                                 norm_layer=norm_layer, with_cp=with_cp)
        self.extractor = Extractor(dim=dim,num_heads=num_heads, norm_layer=norm_layer, with_cffn=with_cffn,
                                   cffn_ratio=cffn_ratio, drop=drop, drop_path=drop_path, with_cp=with_cp)
        
        if extra_extractor:
            self.extra_extractors = nn.Sequential(*[
                Extractor(dim=dim, num_heads=num_heads, norm_layer=norm_layer,
                          with_cffn=with_cffn, cffn_ratio=cffn_ratio,
                          drop=drop, drop_path=drop_path, with_cp=with_cp)
                for _ in range(2)
            ])
        else:
            self.extra_extractors = None
    
    def forward(self, x, c, blocks, num_valid_people):
        x = self.injector(query=x, feat=c)
        for idx, blk in enumerate(blocks):
            x = blk(x)
        c = self.extractor(query=c, feat=x)
        if self.extra_extractors is not None:
            for extractor in self.extra_extractors:
                c = extractor(query=c, feat=x)
        
        return x, c