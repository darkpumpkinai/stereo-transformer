#  Authors: Zhaoshuo Li, Xingtong Liu, Francis X. Creighton, Russell H. Taylor, and Mathias Unberath
#
#  Copyright (c) 2020. Johns Hopkins University - All rights reserved.
#
# Modified by Alex Showalter-Bucher(alex@darkpumpkin.ai)
# -Fixed issues with running batches on a single GPU 05/27/2021
# -Modified to add partial linear attention to reduce memory requirements 06/02/2021

# """
# Linear Transformer proposed in "Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention"
# Modified from:
#  https://github.com/idiap/fast-transformers/blob/master/fast_transformers/attention/linear_attention.py
#  https://github.com/zju3dv/LoFTR
# """
import torch
import torch.nn as nn
import torch.nn.functional as F


def elu_feature_map(x):
    """
        Kernel for linear attention matrix
    """
    return torch.nn.functional.elu(x) + 1


def project(attn_obj, query, key, value, pos_enc, pos_indexes):
    """
    Multihead attention projection logic
        :param attn_obj: an instantiated nn.MultiheadAttention object
        :param query: [W,HN,C]
        :param key: [W,HN,C]
        :param value: [W,HN,C]
        :param pos_enc: [2W-1,C]
        :param pos_indexes: index to select relative encodings, flattened in transformer WW
        :return: output value vector, attention with softmax (for debugging) and raw attention (used for last layer)
    """

    w, bsz, embed_dim = query.size()
    head_dim = embed_dim // attn_obj.num_heads
    assert head_dim * attn_obj.num_heads == embed_dim, "embed_dim must be divisible by num_heads"

    # project to get qkv
    if torch.equal(query, key) and torch.equal(key, value):
        # self-attention
        q, k, v = F.linear(query, attn_obj.in_proj_weight, attn_obj.in_proj_bias).chunk(3, dim=-1)

    elif torch.equal(key, value):
        # cross-attention
        _b = attn_obj.in_proj_bias
        _start = 0
        _end = embed_dim
        _w = attn_obj.in_proj_weight[_start:_end, :]
        if _b is not None:
            _b = _b[_start:_end]
        q = F.linear(query, _w, _b)

        if key is None:
            assert value is None
            k = None
            v = None
        else:
            _b = attn_obj.in_proj_bias
            _start = embed_dim
            _end = None
            _w = attn_obj.in_proj_weight[_start:, :]
            if _b is not None:
                _b = _b[_start:]
            k, v = F.linear(key, _w, _b).chunk(2, dim=-1)

    # project to find q_r, k_r
    if pos_enc is not None:
        # reshape pos_enc
        pos_enc = torch.index_select(pos_enc, 0, pos_indexes).view(w, w,
                                                                   -1)  # 2W-1xC -> WW'xC -> WxW'xC
        # compute k_r, q_r
        _start = 0
        _end = 2 * embed_dim
        _w = attn_obj.in_proj_weight[_start:_end, :]
        _b = attn_obj.in_proj_bias[_start:_end]
        q_r, k_r = F.linear(pos_enc, _w, _b).chunk(2, dim=-1)  # WxW'xC
    else:
        q_r = None
        k_r = None

    # scale query
    scaling = float(head_dim) ** -0.5
    q = q * scaling
    if q_r is not None:
        q_r = q_r * scaling

    # reshape
    q = q.contiguous().view(w, bsz, attn_obj.num_heads, head_dim)  # WxNxExC
    if k is not None:
        k = k.contiguous().view(-1, bsz, attn_obj.num_heads, head_dim)
    if v is not None:
        v = v.contiguous().view(-1, bsz, attn_obj.num_heads, head_dim)

    if q_r is not None:
        q_r = q_r.contiguous().view(w, w, attn_obj.num_heads, head_dim)  # WxW'xExC
    if k_r is not None:
        k_r = k_r.contiguous().view(w, w, attn_obj.num_heads, head_dim)

    return q, k, v, q_r, k_r


class MultiheadLinearAttentionRelative(nn.MultiheadAttention):
    """
    Multihead linear attention with relative positional encoding
    """

    def __init__(self, embed_dim, num_heads, eps=1e-6):
        self.eps = eps

        super(MultiheadLinearAttentionRelative, self).__init__(embed_dim, num_heads, dropout=0.0, bias=True,
                                                         add_bias_kv=False, add_zero_attn=False,
                                                         kdim=None, vdim=None)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.feature_map = elu_feature_map

    def linear_attention(self, q, k, v, bsz, w, head_dim, pos_flag=False):
        """
        Multihead attention

        :param query: [W,HN,C]
        :param key: [W,HN,C]
        :param value: [W,HN,C]
        :param attn_mask: mask to invalidate attention, -inf is used for invalid attention, [W,W]
        :param pos_enc: [2W-1,C]
        :param pos_indexes: index to select relative encodings, flattened in transformer WW
        :return: output value vector, attention with softmax (for debugging) and raw attention (used for last layer)
        """

        # Note the comments in the function were mostly borrowed from the linear attention repos mentioned at the top
        # of this file
        v_length = v.size(0)
        values = v / v_length  # prevent fp16 overflow

        if pos_flag:
            # Compute the KV matrix, namely the dot product of keys and values so
            # that we never explicitly compute the attention matrix and thus
            # decrease the complexity
            kv = torch.einsum("wphd,wnhv->nhdv", k, values)  # (S,D)' @ S,V
            # Compute the normalizer
            z = 1 / (torch.einsum("wnhd,whd->nwh", q, k.sum(dim=0)) + self.eps)
        else:
            # Compute the KV matrix, namely the dot product of keys and values so
            # that we never explicitly compute the attention matrix and thus
            # decrease the complexity
            kv = torch.einsum("wnhd,wnhv->nhdv", k, values)  # (S,D)' @ S,V
            # Compute the normalizer
            z = 1 / (torch.einsum("wnhd,nhd->nwh", q, k.sum(dim=0)) + self.eps)

        # Finally compute and return the new values
        queried_values = torch.einsum("wnhd,nhdv,nwh->wnhv", q, kv, z) * v_length

        v_o_feat = queried_values.contiguous().view(w, bsz, self.num_heads * head_dim)

        return v_o_feat

    def forward(self, query, key, value, attn_mask=None, pos_enc=None, pos_indexes=None):
        """
        Multihead attention

        :param query: [W,HN,C]
        :param key: [W,HN,C]
        :param value: [W,HN,C]
        :param attn_mask: mask to invalidate attention, -inf is used for invalid attention, [W,W]
        :param pos_enc: [2W-1,C]
        :param pos_indexes: index to select relative encodings, flattened in transformer WW
        :return: output value vector, attention with softmax (for debugging) and raw attention (used for last layer)
        """

        w, bsz, embed_dim = query.size()
        head_dim = embed_dim // self.num_heads
        assert head_dim * self.num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        # This projects the qkv / positional information
        q, k, v, q_r, k_r = project(self, query, key, value, pos_enc, pos_indexes)

        # Linear Attention Calculations
        q = self.feature_map(q)
        k = self.feature_map(k)

        v_o_feat = self.linear_attention(q, k, v, bsz, w, head_dim)
        # add positional terms
        if pos_enc is not None:
            # 0.3 s
            q_r = self.feature_map(q_r)
            k_r = self.feature_map(k_r)
            v_o_feat_pos = self.linear_attention(q, k_r, v, bsz, w, head_dim, pos_flag=True)
            v_o_pos_feat = self.linear_attention(k, q_r, v, bsz, w, head_dim, pos_flag=True)

            # This really should have been added as attention weights, but since that intermediate product doesn't exist
            # for linear attention I'm summing them after the full calculations (TODO verify this is okay)
            v_o = v_o_feat + v_o_feat_pos + v_o_pos_feat
        else:
            v_o = v_o_feat

        # TODO verify the self.norm1 is needed in this
        v_o = F.linear(self.norm1(v_o), self.out_proj.weight, self.out_proj.bias)

        # The last layer uses an attn_mask so we need the full attention matrix :-(
        # (TODO figure out a way to remove the need to calculate the full attention matrix)
        if attn_mask is not None:
            # # compute attn weight
            attn_feat = torch.einsum('wnec,vnec->newv', q, k)  # NxExWxW'

            # add positional terms
            if pos_enc is not None:
                # 0.3 s
                attn_feat_pos = torch.einsum('wnec,wvec->newv', q, k_r)  # NxExWxW'
                attn_pos_feat = torch.einsum('vnec,wvec->newv', k, q_r)  # NxExWxW'

                # 0.1 s
                attn = attn_feat + attn_feat_pos + attn_pos_feat
            else:
                attn = attn_feat
            assert list(attn.size()) == [bsz, self.num_heads, w, w]

            # apply attn mask
            attn_mask = attn_mask[None, None, ...]
            attn += attn_mask

            # raw attn
            raw_attn = attn.sum(dim=1)
        else:
            raw_attn = torch.empty(1, 1, dtype=torch.bool)

        return v_o, None, raw_attn


class MultiheadAttentionRelative(nn.MultiheadAttention):
    """
    Multihead attention with relative positional encoding
    """

    def __init__(self, embed_dim, num_heads):
        super(MultiheadAttentionRelative, self).__init__(embed_dim, num_heads, dropout=0.0, bias=True,
                                                         add_bias_kv=False, add_zero_attn=False,
                                                         kdim=None, vdim=None)

    def forward(self, query, key, value, attn_mask=None, pos_enc=None, pos_indexes=None):
        """
        Multihead attention

        :param query: [W,HN,C]
        :param key: [W,HN,C]
        :param value: [W,HN,C]
        :param attn_mask: mask to invalidate attention, -inf is used for invalid attention, [W,W]
        :param pos_enc: [2W-1,C]
        :param pos_indexes: index to select relative encodings, flattened in transformer WW
        :return: output value vector, attention with softmax (for debugging) and raw attention (used for last layer)
        """

        w, bsz, embed_dim = query.size()
        head_dim = embed_dim // self.num_heads
        assert head_dim * self.num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        # This projects the qkv / positional information
        q, k, v, q_r, k_r = project(self, query, key, value, pos_enc, pos_indexes)

        # compute attn weight
        attn_feat = torch.einsum('wnec,vnec->newv', q, k)  # NxExWxW'

        # add positional terms
        if pos_enc is not None:
            # 0.3 s
            attn_feat_pos = torch.einsum('wnec,wvec->newv', q, k_r)  # NxExWxW'
            attn_pos_feat = torch.einsum('vnec,wvec->newv', k, q_r)  # NxExWxW'

            # 0.1 s
            attn = attn_feat + attn_feat_pos + attn_pos_feat
        else:
            attn = attn_feat

        assert list(attn.size()) == [bsz, self.num_heads, w, w]

        # apply attn mask
        if attn_mask is not None:
            attn_mask = attn_mask[None, None, ...]
            attn += attn_mask

        # raw attn
        raw_attn = attn

        # softmax
        attn = F.softmax(attn, dim=-1)

        # compute v, equivalent to einsum('',attn,v),
        # need to do this because apex does not support einsum when precision is mixed
        v_o = torch.bmm(attn.view(bsz * self.num_heads, w, w),
                        v.permute(1, 2, 0, 3).view(bsz * self.num_heads, w, head_dim))  # NxExWxW', W'xNxExC -> NExWxC
        assert list(v_o.size()) == [bsz * self.num_heads, w, head_dim]
        v_o = v_o.reshape(bsz, self.num_heads, w, head_dim).permute(2, 0, 1, 3).reshape(w, bsz, embed_dim)
        v_o = F.linear(v_o, self.out_proj.weight, self.out_proj.bias)

        # average attention weights over heads
        attn = attn.sum(dim=1) / self.num_heads

        # raw attn
        raw_attn = raw_attn.sum(dim=1)

        return v_o, attn, raw_attn
