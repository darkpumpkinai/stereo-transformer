import copy
import torch
import torch.nn as nn
from .linear_attention import LinearAttention, FullAttention
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
from .position_encoding import PositionEncodingSine1D


class LoFTREncoderLayer(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 attention='linear'):
        super(LoFTREncoderLayer, self).__init__()

        self.dim = d_model // nhead
        self.nhead = nhead
        self.pos_encoding = PositionEncodingSine1D(d_model)
        # multi-head attention
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.attention = LinearAttention() if attention == 'linear' else FullAttention()
        self.merge = nn.Linear(d_model, d_model, bias=False)

        # feed-forward network
        self.mlp = nn.Sequential(
            nn.Linear(d_model*2, d_model*2, bias=False),
            nn.ReLU(True),
            nn.Linear(d_model*2, d_model, bias=False),
        )

        # norm and dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, source, pos_encoding, x_mask=None, source_mask=None):
        """
        Args:
            x (torch.Tensor): [N, L, C]
            source (torch.Tensor): [N, S, C]
            pos_encoding: (torch.model)  function
            x_mask (torch.Tensor): [N, L] (optional)
            source_mask (torch.Tensor): [N, S] (optional)
        """


        # 2. coarse-level loftr module
        # add featmap with positional encoding, then flatten it to sequence [N, HW, C]
        x = self.pos_encoding(x)
        source = self.pos_encoding(source)
        bs, h, w, c = x.size()
        query, key, value = x, source, source

        # multi-head attention
        query = self.q_proj(query).view(bs, h, w, self.nhead, self.dim)  # [N, L, (H, D)]
        key = self.k_proj(key).view(bs, h, w, self.nhead, self.dim)  # [N, S, (H, D)]
        value = self.v_proj(value).view(bs, h, w, self.nhead, self.dim)
        message = self.attention(query, key, value, q_mask=x_mask, kv_mask=source_mask)  # [N, L, (H, D)]
        message = self.merge(message.view(bs, h, w, self.nhead*self.dim))  # [N, L, C]
        message = self.norm1(message)

        # feed-forward network
        message = self.mlp(torch.cat([x, message], dim=3))
        message = self.norm2(message)

        return x + message


class LocalFeatureTransformer(nn.Module):
    """A Local Feature Transformer (LoFTR) module."""

    def __init__(self, config):
        super(LocalFeatureTransformer, self).__init__()

        self.config = config
        self.d_model = config['d_model']
        self.nhead = config['nhead']
        self.layer_names = config['layer_names']
        self.temperature = 5 #0.1
        encoder_layer = LoFTREncoderLayer(config['d_model'], config['nhead'], config['attention'])

        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(len(self.layer_names))])
        self._reset_parameters()

    @torch.no_grad()
    def _generate_square_subsequent_mask(self, sz: int):
        """
        Generate a mask which is upper triangular

        :param sz: square matrix size
        :return: diagonal binary mask [sz,sz]
        """
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask[mask == 1] = float('-inf')
        return mask

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, feat0, feat1,mask0=None, mask1=None ):
        """
        Args:
            feat0 (torch.Tensor): [N, L, C]
            feat1 (torch.Tensor): [N, S, C]
            mask0 (torch.Tensor): [N, L] (optional)
            mask1 (torch.Tensor): [N, S] (optional)
        """
        mask0 = None
        mask1 = None
        feat0 = feat0.permute(0, 2, 3, 1)
        feat1 = feat1.permute(0, 2, 3, 1)

        assert self.d_model == feat0.size(3), "the feature number of src and transformer must be equal"

        for layer, name in zip(self.layers, self.layer_names):
            if name == 'self':
                feat0 = layer(feat0, feat0, mask0, mask0)
                feat1 = layer(feat1, feat1, mask1, mask1)
            elif name == 'cross':
                feat0 = layer(feat0, feat1, mask0, mask1)
                feat1 = layer(feat1, feat0, mask1, mask0)
            else:
                raise KeyError

        # Set raw_attn to be a similarity matrix as an approximation for the linear attention
        w = feat0.size(2)
        attn_mask = self._generate_square_subsequent_mask(w).to(feat0.device)  # generate attn mask
        attn_mask = attn_mask[None, None, ...]
        # normalize
        feat_c0, feat_c1 = map(lambda feat: feat / feat.shape[-1]**.5, [feat0, feat1])
        sim_matrix = torch.einsum("ntwc,ntlc->ntwl", feat_c0, feat_c1) / self.temperature
        raw_attn = sim_matrix + attn_mask
        vis_raw_attn = np.amax(raw_attn.squeeze().detach().cpu().numpy(), axis=2)
        return raw_attn