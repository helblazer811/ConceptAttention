from collections import OrderedDict
import math
from typing import Callable, Optional, Sequence, Tuple, Text

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
import numbers
import einops
import numpy as np
from concept_attention.binary_segmentation_baselines.clip_text_span.utils.misc import to_2tuple
from concept_attention.binary_segmentation_baselines.clip_text_span.utils.hook import HookManager


class LayerNorm(nn.Module):
    """Subclass torch's LayerNorm (with cast back to input dtype)."""

    def __init__(
        self,
        normalized_shape,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        device=None,
        dtype=None,
        hook: Optional[HookManager] = None,
    ):
        super().__init__()
        self.hook = hook or HookManager()
        if isinstance(normalized_shape, numbers.Integral):
            # mypy error: incompatible types in assignment
            normalized_shape = (normalized_shape,)  # type: ignore[assignment]
        self.normalized_shape = tuple(normalized_shape)  # type: ignore[arg-type]
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = torch.nn.Parameter(
                torch.empty(
                    self.normalized_shape,
                )
            )
            self.bias = torch.nn.Parameter(
                torch.empty(
                    self.normalized_shape,
                )
            )
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        assert self.normalized_shape == x.shape[-len(self.normalized_shape) :]
        dims = [-(i + 1) for i in range(len(self.normalized_shape))]
        mean = self.hook("mean", ret=x.mean(dim=dims, keepdim=True))
        mean_x2 = (x**2).mean(dim=dims, keepdim=True)
        var = mean_x2 - mean**2
        x_norm = self.hook("mean_reduced", ret=(x - mean)) / self.hook(
            "sqrt_var", ret=torch.sqrt(var + self.eps)
        )
        if self.elementwise_affine:
            x_norm = self.hook("renorm.post", ret=self.weight * x_norm + self.bias)
        self.hook.finalize()
        return x_norm.to(orig_type)


class QuickGELU(nn.Module):
    # NOTE This is slower than nn.GELU or nn.SiLU and uses more GPU memory
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        raise ValueError("Not implemented")
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class PatchDropout(nn.Module):
    """
    https://arxiv.org/abs/2212.00794
    """

    def __init__(self, prob, exclude_first_token=True):
        super().__init__()
        assert 0 <= prob < 1.0
        self.prob = prob
        self.exclude_first_token = exclude_first_token  # exclude CLS token

    def forward(self, x):
        if not self.training or self.prob == 0.0:
            return x

        if self.exclude_first_token:
            cls_tokens, x = x[:, :1], x[:, 1:]
        else:
            cls_tokens = torch.jit.annotate(torch.Tensor, x[:, :1])

        batch = x.size()[0]
        num_tokens = x.size()[1]

        batch_indices = torch.arange(batch)
        batch_indices = batch_indices[..., None]

        keep_prob = 1 - self.prob
        num_patches_keep = max(1, int(num_tokens * keep_prob))

        rand = torch.randn(batch, num_tokens)
        patch_indices_keep = rand.topk(num_patches_keep, dim=-1).indices

        x = x[batch_indices, patch_indices_keep]

        if self.exclude_first_token:
            x = torch.cat((cls_tokens, x), dim=1)

        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=True,
        scaled_cosine=False,
        scale_heads=False,
        logit_scale_max=math.log(1.0 / 0.01),
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.scaled_cosine = scaled_cosine
        self.scale_heads = scale_heads
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.logit_scale_max = logit_scale_max

        # keeping in_proj in this form (instead of nn.Linear) to match weight scheme of original
        self.in_proj_weight = nn.Parameter(torch.randn((dim * 3, dim)) * self.scale)
        if qkv_bias:
            self.in_proj_bias = nn.Parameter(torch.zeros(dim * 3))
        else:
            self.in_proj_bias = None

        if self.scaled_cosine:
            self.logit_scale = nn.Parameter(
                torch.log(10 * torch.ones((num_heads, 1, 1)))
            )
        else:
            self.logit_scale = None
        self.attn_drop = nn.Dropout(attn_drop)
        if self.scale_heads:
            self.head_scale = nn.Parameter(torch.ones((num_heads, 1, 1)))
        else:
            self.head_scale = None
        self.out_proj = nn.Linear(dim, dim)
        self.out_drop = nn.Dropout(proj_drop)

    def forward(self, x, attn_mask: Optional[torch.Tensor] = None):
        L, N, C = x.shape
        q, k, v = F.linear(x, self.in_proj_weight, self.in_proj_bias).chunk(3, dim=-1)
        q = q.contiguous().view(L, N * self.num_heads, -1).transpose(0, 1)
        k = k.contiguous().view(L, N * self.num_heads, -1).transpose(0, 1)
        v = v.contiguous().view(L, N * self.num_heads, -1).transpose(0, 1)

        if self.logit_scale is not None:
            attn = torch.bmm(
                F.normalize(q, dim=-1), F.normalize(k, dim=-1).transpose(-1, -2)
            )
            logit_scale = torch.clamp(self.logit_scale, max=self.logit_scale_max).exp()
            attn = attn.view(N, self.num_heads, L, L) * logit_scale
            attn = attn.view(-1, L, L)
        else:
            q = q * self.scale
            attn = torch.bmm(q, k.transpose(-1, -2))

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                new_attn_mask = torch.zeros_like(attn_mask, dtype=q.dtype)
                new_attn_mask.masked_fill_(attn_mask, float("-inf"))
                attn_mask = new_attn_mask
            attn += attn_mask

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = torch.bmm(attn, v)
        if self.head_scale is not None:
            x = x.view(N, self.num_heads, L, C) * self.head_scale
            x = x.view(-1, L, C)
        x = x.transpose(0, 1).reshape(L, N, C)
        x = self.out_proj(x)
        x = self.out_drop(x)
        return x


class AttentionalPooler(nn.Module):
    def __init__(
        self,
        d_model: int,
        context_dim: int,
        n_head: int = 8,
        n_queries: int = 256,
        norm_layer: Callable = LayerNorm,
    ):
        super().__init__()
        self.query = nn.Parameter(torch.randn(n_queries, d_model))
        self.attn = nn.MultiheadAttention(
            d_model, n_head, kdim=context_dim, vdim=context_dim
        )
        self.ln_q = norm_layer(d_model)
        self.ln_k = norm_layer(context_dim)

    def forward(self, x: torch.Tensor):
        x = self.ln_k(x).permute(1, 0, 2)  # NLD -> LND
        N = x.shape[1]
        q = self.ln_q(self.query)
        out = self.attn(self._repeat(q, N), x, x, need_weights=False)[0]
        return out.permute(1, 0, 2)  # LND -> NLD

    def _repeat(self, query, N: int):
        return query.unsqueeze(1).repeat(1, N, 1)


class MLP(nn.Module):
    def __init__(
        self,
        d_model: int,
        mlp_width: int,
        act_layer: Callable = nn.GELU,
        hook: Optional[HookManager] = None,
    ):
        super().__init__()
        self.hook = hook or HookManager()
        self.c_fc = nn.Linear(d_model, mlp_width)
        self.gelu = act_layer()
        self.c_proj = nn.Linear(mlp_width, d_model)

    def forward(self, x):
        x = self.hook("c_fc.post", ret=self.c_fc(x))
        x = self.hook("gelu.post", ret=self.gelu(x))
        x = self.hook("c_proj.post", ret=self.c_proj(x))
        self.hook.finalize()
        return x


class MultiheadAttention(nn.Module):
    """
    There are variety of ways to look at multihead attention. Because of that I implemented a few so it will be easy to compare.
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
        bias=True,
        add_bias_kv=False,
        add_zero_attn=False,
        kdim=None,
        vdim=None,
        batch_first=False,
        device=None,
        dtype=None,
        hook: Optional[HookManager] = None,
    ):
        super().__init__()
        self.hook = hook or HookManager()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"
        self.in_proj_weight = nn.Parameter(torch.empty((3 * embed_dim, embed_dim)))

        if bias:
            self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim))
        else:
            self.register_parameter("in_proj_bias", None)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        if add_bias_kv:
            self.bias_k = nn.Parameter(torch.empty((1, 1, embed_dim)))
            self.bias_v = nn.Parameter(torch.empty((1, 1, embed_dim)))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

    def forward_direct(self, x, attn_mask=None):
        B, N, C = x.shape
        qkv = self.hook(
            "in_proj_bias.post",
            ret=self.hook("in_proj.post", ret=x @ self.in_proj_weight.T)
            + self.in_proj_bias,
        )
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        k = self.hook("k", ret=k)
        q = self.hook("q", ret=q)
        v = self.hook("v", ret=v)
        dk = q.size()[-1]
        q = q / math.sqrt(dk)
        q = self.hook("q_norm", ret=q)
        attn = q @ k.transpose(-2, -1)  # [B, H, N, N]
        attn = self.hook("pre_mask", ret=attn)
        if attn_mask is not None:
            attn += attn_mask
        attn = self.hook("post_mask", ret=attn)
        attn = attn.softmax(dim=-1)
        attn = self.hook("post_softmax", ret=attn)
        x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.hook("attn_v", ret=x)
        x = self.hook(
            "out_proj_bias.post",
            ret=self.hook("out_proj.post", ret=x @ self.out_proj.weight.T)
            + self.out_proj.bias,
        )
        return x

    def _split_qkv_weight(self):
        q_weight, k_weight, v_weight = (
            self.in_proj_weight[: self.embed_dim].reshape(
                self.num_heads, self.head_dim, -1
            ),
            self.in_proj_weight[self.embed_dim : self.embed_dim * 2].reshape(
                self.num_heads, self.head_dim, -1
            ),
            self.in_proj_weight[self.embed_dim * 2 :].reshape(
                self.num_heads, self.head_dim, -1
            ),
        )
        return q_weight, k_weight, v_weight

    def _split_qkv_bias(self):
        q_bias, k_bias, v_bias = (
            self.in_proj_bias[: self.embed_dim].reshape(
                1, self.num_heads, 1, self.head_dim
            ),
            self.in_proj_bias[self.embed_dim : self.embed_dim * 2].reshape(
                1, self.num_heads, 1, self.head_dim
            ),
            self.in_proj_bias[self.embed_dim * 2 :].reshape(
                1, self.num_heads, 1, self.head_dim
            ),
        )
        return q_bias, k_bias, v_bias

    def forward_qkv(self, x, attn_mask=None):
        B, N, C = x.shape
        q_weight, k_weight, v_weight = (
            self.in_proj_weight[: self.embed_dim],
            self.in_proj_weight[self.embed_dim : self.embed_dim * 2],
            self.in_proj_weight[self.embed_dim * 2 :],
        )
        q_bias, k_bias, v_bias = (
            self.in_proj_bias[: self.embed_dim],
            self.in_proj_bias[self.embed_dim : self.embed_dim * 2],
            self.in_proj_bias[self.embed_dim * 2 :],
        )
        q = (
            self.hook(
                "in_q_bias.post",
                ret=self.hook("in_q.post", ret=x @ q_weight.T) + q_bias,
            )
            .reshape(B, N, self.num_heads, self.head_dim)
            .permute(0, 2, 1, 3)
        )
        k = (
            self.hook(
                "in_k_bias.post",
                ret=self.hook("in_k.post", ret=x @ k_weight.T) + k_bias,
            )
            .reshape(B, N, self.num_heads, self.head_dim)
            .permute(0, 2, 1, 3)
        )
        v = (
            self.hook(
                "in_v_bias.post",
                ret=self.hook("in_v.post", ret=x @ v_weight.T) + v_bias,
            )
            .reshape(B, N, self.num_heads, self.head_dim)
            .permute(0, 2, 1, 3)
        )
        dk = q.size()[-1]
        q = q / math.sqrt(dk)
        q = self.hook("q_norm", ret=q)
        attn = q @ k.transpose(-2, -1)
        attn = self.hook("attention.pre_mask", ret=attn)
        if attn_mask is not None:
            attn += attn_mask
        attn = self.hook("attention.post_mask", ret=attn)
        attn = attn.softmax(dim=-1)
        attn = self.hook("attention.post_softmax", ret=attn)  # [B, H, N, N]
        x = torch.einsum("bhnm,bhmc->bhnmc", attn, v)
        x = self.hook("extended_attn_v", ret=x)
        x = x.sum(axis=3).transpose(1, 2).reshape(B, N, C)
        x = self.hook("attn_v", ret=x)
        x = self.hook(
            "out.post_bias",
            ret=self.hook("out.post", ret=x @ self.out_proj.weight.T)
            + self.out_proj.bias,
        )
        return x

    def forward_per_head_no_spatial(self, x, attn_mask=None):
        B, N, C = x.shape
        q_weight, k_weight, v_weight = self._split_qkv_weight()
        q_bias, k_bias, v_bias = self._split_qkv_bias()
        q = self.hook(
            "in_q_bias.post",
            ret=self.hook("in_q.post", ret=torch.einsum("bnc,hdc->bhnd", x, q_weight))
            + q_bias,
        )
        k = self.hook(
            "in_k_bias.post",
            ret=self.hook("in_k.post", ret=torch.einsum("bnc,hdc->bhnd", x, k_weight))
            + k_bias,
        )
        v = self.hook(
            "in_v_bias.post",
            ret=self.hook("in_v.post", ret=torch.einsum("bnc,hdc->bhnd", x, v_weight))
            + v_bias,
        )  # (B, self.num_heads, N, self.head_dim)
        dk = q.size()[-1]
        q = q / math.sqrt(dk)
        q = self.hook("q_norm", ret=q)
        attn = q @ k.transpose(-2, -1)
        attn = self.hook("attention.pre_mask", ret=attn)
        if attn_mask is not None:
            attn += attn_mask
        attn = self.hook("attention.post_mask", ret=attn)
        attn = attn.softmax(dim=-1)
        attn = self.hook("attention.post_softmax", ret=attn)  # [B, H, N, N]
        x = torch.einsum(
            "bhnm,bhmc->bnhc", attn, v
        )  # We also switch here back from head-first to n-first
        x = self.hook("attn_v", ret=x)
        x = self.hook(
            "out.post",
            ret=torch.einsum(
                "bnhc,dhc->bnhd",
                x,
                self.out_proj.weight.reshape(
                    self.embed_dim, self.num_heads, self.head_dim
                ),
            ),
        )
        x = self.hook("out.post_collapse", ret=x.sum(axis=2))
        x = self.hook("out.post_bias", ret=x + self.out_proj.bias)
        return x


    def forward_per_head(self, x, attn_mask=None):
        B, N, C = x.shape
        q_weight, k_weight, v_weight = self._split_qkv_weight()
        q_bias, k_bias, v_bias = self._split_qkv_bias()
        q = self.hook(
            "in_q_bias.post",
            ret=self.hook("in_q.post", ret=torch.einsum("bnc,hdc->bhnd", x, q_weight))
            + q_bias,
        )
        k = self.hook(
            "in_k_bias.post",
            ret=self.hook("in_k.post", ret=torch.einsum("bnc,hdc->bhnd", x, k_weight))
            + k_bias,
        )
        v = self.hook(
            "in_v_bias.post",
            ret=self.hook("in_v.post", ret=torch.einsum("bnc,hdc->bhnd", x, v_weight))
            + v_bias,
        )  # (B, self.num_heads, N, self.head_dim)
        dk = q.size()[-1]
        q = q / math.sqrt(dk)
        q = self.hook("q_norm", ret=q)
        attn = q @ k.transpose(-2, -1)
        attn = self.hook("attention.pre_mask", ret=attn)
        if attn_mask is not None:
            attn += attn_mask
        attn = self.hook("attention.post_mask", ret=attn)
        attn = attn.softmax(dim=-1)
        attn = self.hook("attention.post_softmax", ret=attn)  # [B, H, N, N]
        x = torch.einsum(
            "bhnm,bhmc->bnmhc", attn, v
        )  # We also switch here back from head-first to n-first
        x = self.hook("extended_attn_v", ret=x)
        x = self.hook(
            "out.post",
            ret=torch.einsum(
                "bnmhc,dhc->bnmhd",
                x,
                self.out_proj.weight.reshape(
                    self.embed_dim, self.num_heads, self.head_dim
                ),
            ),
        )
        x = self.hook("out.post_collapse", ret=x.sum(axis=[2, 3]))
        x = self.hook("out.post_bias", ret=x + self.out_proj.bias)
        return x

    def _get_ov_circuit(
        self,
    ):
        reshaped_o = self.out_proj.weight.reshape(
            self.embed_dim, self.num_heads, self.head_dim
        )
        _, _, v_weight = self._split_qkv_weight()  # num_heads, head_dim, embed_dim
        _, _, v_bias = self._split_qkv_bias()  # 1, num_heads, 1, head_dim
        ov_circuit = torch.einsum("onh,nhi->oni", reshaped_o, v_weight)
        ov_bias_circuit = torch.einsum(
            "onh,bnxh->bnxo", reshaped_o, v_bias
        )  # [1, num_heads, 1, embed_dim]
        return ov_circuit, ov_bias_circuit

    def forward_ov_circuit(self, x, attn_mask=None):
        B, N, C = x.shape
        q_weight, k_weight, _ = self._split_qkv_weight()
        q_bias, k_bias, _ = self._split_qkv_bias()
        q = self.hook(
            "in_q_bias.post",
            ret=self.hook("in_q.post", ret=torch.einsum("bnc,hdc->bhnd", x, q_weight))
            + q_bias,
        )
        k = self.hook(
            "in_k_bias.post",
            ret=self.hook("in_k.post", ret=torch.einsum("bnc,hdc->bhnd", x, k_weight))
            + k_bias,
        )
        ov, ov_bias = self._get_ov_circuit()
        ov = self.hook("ov", ret=ov)
        ov_bias = self.hook("ov_bias", ret=ov_bias)
        v = self.hook(
            "ov_bias.post",
            ret=self.hook("ov.post", ret=torch.einsum("bnc,dhc->bhnd", x, ov))
            + ov_bias,
        )

        dk = q.size()[-1]
        q = q / math.sqrt(dk)
        q = self.hook("q_norm", ret=q)
        attn = q @ k.transpose(-2, -1)
        attn = self.hook("attention.pre_mask", ret=attn)
        if attn_mask is not None:
            attn += attn_mask
        attn = self.hook("attention.post_mask", ret=attn)
        attn = attn.softmax(dim=-1)
        attn = self.hook("attention.post_softmax", ret=attn)  # [B, H, N, N]
        x = torch.einsum(
            "bhnm,bhmc->bnmhc", attn, v
        )  # We also switch here back from head-first to n-first
        x = self.hook("extended_attn_ov", ret=x)
        x = self.hook("out.post_collapse", ret=x.sum(axis=[2, 3]))
        x = self.hook("out.post_bias", ret=x + self.out_proj.bias)
        return x

    def forward(self, x, attn_mask=None, method: Text = "ov_circuit"):
        if method == "direct":
            x = self.forward_direct(x, attn_mask=attn_mask)
        elif method == "qkv":
            x = self.forward_qkv(x, attn_mask=attn_mask)
        elif method == "head":
            x = self.forward_per_head(x, attn_mask=attn_mask)
        elif method == "head_no_spatial":
            x = self.forward_per_head_no_spatial(x, attn_mask=attn_mask)
        elif method == "ov_circuit":
            x = self.forward_ov_circuit(x, attn_mask=attn_mask)
        else:
            raise NotImplementedError('Unknown attention method')
        self.hook.finalize()

        return x


class ResidualAttentionBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_head: int,
        mlp_ratio: float = 4.0,
        ls_init_value: float = None,
        act_layer: Callable = nn.GELU,
        norm_layer: Callable = LayerNorm,
        hook: Optional[HookManager] = None,
    ):
        super().__init__()
        self.hook = hook or HookManager()
        self.ln_1 = norm_layer(d_model, hook=hook.fork("ln_1"))
        self.attn = MultiheadAttention(d_model, n_head, hook=hook.fork("attn"))

        self.ls_1 = (
            LayerScale(d_model, ls_init_value)
            if ls_init_value is not None
            else nn.Identity()
        )

        self.ln_2 = norm_layer(d_model, hook=hook.fork("ln_2"))
        mlp_width = int(d_model * mlp_ratio)
        self.mlp = MLP(d_model, mlp_width, act_layer=act_layer, hook=hook.fork("mlp"))
        self.ls_2 = (
            LayerScale(d_model, ls_init_value)
            if ls_init_value is not None
            else nn.Identity()
        )

    def attention(
        self,
        q_x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        method: Text = "direct",
    ):
        attn_mask = attn_mask.to(q_x.dtype) if attn_mask is not None else None
        return self.attn(q_x, attn_mask=attn_mask, method=method)

    def forward(
        self,
        q_x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        attn_method: Text = "direct",
    ):
        q_x = self.hook("pre", ret=q_x)
        after_ln1 = self.ln_1(q_x)
        after_attn = self.attention(
            q_x=after_ln1, attn_mask=attn_mask, method=attn_method
        )
        after_attn = self.hook("after_attn", ret=after_attn)
        x = q_x + self.ls_1(after_attn)
        after_ln2 = self.ln_2(x)
        after_mlp = self.mlp(after_ln2)
        after_mlp = self.hook("after_mlp", ret=after_mlp)
        x = x + self.ls_2(after_mlp)
        x = self.hook("post", ret=x)
        self.hook.finalize()
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        width: int,
        layers: int,
        heads: int,
        mlp_ratio: float = 4.0,
        ls_init_value: float = None,
        act_layer: Callable = nn.GELU,
        norm_layer: Callable = LayerNorm,
        hook: Optional[HookManager] = None,
    ):
        super().__init__()
        self.hook = hook or HookManager()
        self.width = width
        self.layers = layers
        self.grad_checkpointing = False

        self.resblocks = nn.ModuleList(
            [
                ResidualAttentionBlock(
                    width,
                    heads,
                    mlp_ratio,
                    ls_init_value=ls_init_value,
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    hook=hook.fork(f"resblocks.{i}"),
                )
                for i in range(layers)
            ]
        )

    def get_cast_dtype(self) -> torch.dtype:
        if hasattr(self.resblocks[0].mlp.c_fc, "int8_original_dtype"):
            return self.resblocks[0].mlp.c_fc.int8_original_dtype
        return self.resblocks[0].mlp.c_fc.weight.dtype

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        attn_method: Text = "direct",
    ):
        for r in self.resblocks:
            if self.grad_checkpointing and not torch.jit.is_scripting():
                raise ValueError("grad_checkpointing not implement")
                # TODO: handle kwargs https://github.com/pytorch/pytorch/issues/79887#issuecomment-1161758372
                x = checkpoint(r, x, None, None, attn_mask)
            else:
                x = r(x, attn_mask=attn_mask, attn_method=attn_method)
        self.hook.finalize()
        return x


class VisionTransformer(nn.Module):
    output_tokens: torch.jit.Final[bool]

    def __init__(
        self,
        image_size: int,
        patch_size: int,
        width: int,
        layers: int,
        heads: int,
        mlp_ratio: float,
        ls_init_value: float = None,
        global_average_pool: bool = False,
        attentional_pool: bool = False,
        n_queries: int = 256,
        attn_pooler_heads: int = 8,
        output_dim: int = 512,
        patch_dropout: float = 0.0,
        input_patchnorm: bool = False,
        act_layer: Callable = nn.GELU,
        norm_layer: Callable = LayerNorm,
        output_tokens: bool = False,
        hook: Optional[HookManager] = None,
    ):
        super().__init__()
        self.hook = hook or HookManager()
        self.output_tokens = output_tokens
        image_height, image_width = self.image_size = to_2tuple(image_size)
        patch_height, patch_width = self.patch_size = to_2tuple(patch_size)
        self.grid_size = (image_height // patch_height, image_width // patch_width)
        self.output_dim = output_dim

        # whether to layernorm each patch, as done in dual patchnorm paper - https://arxiv.org/abs/2302.01327v1
        self.input_patchnorm = input_patchnorm

        if input_patchnorm:
            patch_input_dim = patch_height * patch_width * 3
            self.patchnorm_pre_ln = LayerNorm(
                patch_input_dim, hook=hook.fork("patchnorm_pre_ln")
            )
            self.conv1 = nn.Linear(patch_input_dim, width)
        else:
            self.patchnorm_pre_ln = nn.Identity()
            self.conv1 = nn.Conv2d(
                in_channels=3,
                out_channels=width,
                kernel_size=patch_size,
                stride=patch_size,
                bias=False,
            )

        # class embeddings and positional embeddings
        scale = width**-0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(
            scale * torch.randn(self.grid_size[0] * self.grid_size[1] + 1, width)
        )

        # setting a patch_dropout of 0. would mean it is disabled and this function would be the identity fn
        self.patch_dropout = (
            PatchDropout(patch_dropout) if patch_dropout > 0.0 else nn.Identity()
        )

        self.ln_pre = norm_layer(width, hook=hook.fork("ln_pre"))
        self.transformer = Transformer(
            width,
            layers,
            heads,
            mlp_ratio,
            ls_init_value=ls_init_value,
            act_layer=act_layer,
            norm_layer=norm_layer,
            hook=hook.fork("transformer"),
        )

        self.global_average_pool = global_average_pool
        if attentional_pool:
            self.attn_pool = AttentionalPooler(
                output_dim, width, n_head=attn_pooler_heads, n_queries=n_queries
            )
            self.ln_post = norm_layer(output_dim, hook=hook.fork("ln_post"))
            self.proj = nn.Parameter(scale * torch.randn(output_dim, output_dim))
        else:
            self.attn_pool = None
            self.ln_post = norm_layer(width, hook=hook.fork("ln_post"))
            self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.transformer.grad_checkpointing = enable

    def _global_pool(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.global_average_pool:
            return x.mean(dim=1), x
        else:
            return x[:, 0], x[:, 1:]

    def forward(self, x: torch.Tensor, attn_method: Text = "direct"):

        # to patches - whether to use dual patchnorm - https://arxiv.org/abs/2302.01327v1
        if self.input_patchnorm:
            # einops - rearrange(x, 'b c (h p1) (w p2) -> b (h w) (c p1 p2)')
            x = x.reshape(
                x.shape[0],
                x.shape[1],
                self.grid_size[0],
                self.patch_size[0],
                self.grid_size[1],
                self.patch_size[1],
            )
            x = x.permute(0, 2, 4, 1, 3, 5)
            x = x.reshape(x.shape[0], self.grid_size[0] * self.grid_size[1], -1)
            x = self.hook("patchnorm_pre_ln.post", ret=self.patchnorm_pre_ln(x))
            x = self.hook("conv1.post", ret=self.conv1(x))
        else:
            x = self.hook(
                "conv1.post", ret=self.conv1(x)
            )  # shape = [*, width, grid, grid]
            x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
            x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]

        # class embeddings and positional embeddings
        x = torch.cat(
            [
                self.class_embedding.to(x.dtype)
                + torch.zeros(
                    x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
                ),
                x,
            ],
            dim=1,
        )  # shape = [*, grid ** 2 + 1, width]
        x = self.hook(
            "positional_embedding.post", ret=x + self.positional_embedding.to(x.dtype)
        )

        # a patch_dropout of 0. would mean it is disabled and this function would do nothing but return what was passed in
        x = self.hook("patch_dropout.post", ret=self.patch_dropout(x))
        x = self.hook("ln_pre_post", ret=self.ln_pre(x))
        # x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x, attn_method=attn_method)
        # x = x.permute(1, 0, 2)  # LND -> NLD
        if self.attn_pool is not None:
            x = self.hook("attn_pool.post", ret=self.attn_pool(x))
            x = self.hook("ln_post_post", ret=self.ln_post(x))
            pooled, tokens = self.hook("global_pool.post", ret=self._global_pool(x))
        else:
            pooled, tokens = self.hook("global_pool.post", ret=self._global_pool(x))
            pooled = self.hook("ln_post_post", ret=self.ln_post(pooled))

        if self.proj is not None:
            pooled = self.hook(
                "proj.post", ret=self.hook("proj.pre", ret=pooled) @ self.proj
            )

        self.hook.finalize()

        if self.output_tokens:
            return pooled, tokens

        return pooled


class TextTransformer(nn.Module):
    output_tokens: torch.jit.Final[bool]

    def __init__(
        self,
        context_length: int = 77,
        vocab_size: int = 49408,
        width: int = 512,
        heads: int = 8,
        layers: int = 12,
        ls_init_value: float = None,
        output_dim: int = 512,
        act_layer: Callable = nn.GELU,
        norm_layer: Callable = LayerNorm,
        embed_cls: bool = False,
        pad_id: int = 0,
        output_tokens: bool = False,
        hook: Optional[HookManager] = None,
    ):
        super().__init__()
        self.hook = hook or HookManager()
        self.output_tokens = output_tokens
        self.num_pos = self.context_length = context_length
        self.vocab_size = vocab_size
        self.width = width
        self.output_dim = output_dim
        self.heads = heads
        self.pad_id = pad_id

        self.text_projection = nn.Parameter(torch.empty(width, output_dim))

        if embed_cls:
            self.cls_emb = nn.Parameter(torch.empty(width))
            self.num_pos += 1
        else:
            self.cls_emb = None

        self.token_embedding = nn.Embedding(vocab_size, width)
        self.positional_embedding = nn.Parameter(torch.empty(self.num_pos, width))
        self.transformer = Transformer(
            width=width,
            layers=layers,
            heads=heads,
            ls_init_value=ls_init_value,
            act_layer=act_layer,
            norm_layer=norm_layer,
            hook=self.hook.fork("transformer"),
        )
        self.ln_final = norm_layer(width)

        self.register_buffer("attn_mask", self.build_attention_mask(), persistent=False)

        self.init_parameters()

    def init_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)
        if self.cls_emb is not None:
            nn.init.normal_(self.cls_emb, std=0.01)

        proj_std = (self.transformer.width**-0.5) * (
            (2 * self.transformer.layers) ** -0.5
        )
        attn_std = self.transformer.width**-0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width**-0.5)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.transformer.grad_checkpointing = enable

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.num_pos, self.num_pos)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def build_cls_mask(self, text, cast_dtype: torch.dtype):
        cls_mask = (text != self.pad_id).unsqueeze(1)
        cls_mask = F.pad(cls_mask, (1, 0, cls_mask.shape[2], 0), value=1.0)
        additive_mask = torch.empty(
            cls_mask.shape, dtype=cast_dtype, device=cls_mask.device
        )
        additive_mask.fill_(0)
        additive_mask.masked_fill_(~cls_mask, float("-inf"))
        additive_mask = torch.repeat_interleave(additive_mask, self.heads, 0)
        return additive_mask

    def _repeat(self, t, N: int):
        return t.reshape(1, 1, -1).repeat(N, 1, 1)

    def forward(self, text, attn_method: Text = "direct"):
        cast_dtype = self.transformer.get_cast_dtype()
        seq_len = text.shape[1]

        x = self.token_embedding(text).to(cast_dtype)  # [batch_size, n_ctx, d_model]
        attn_mask = self.attn_mask
        if self.cls_emb is not None:
            seq_len += 1
            x = torch.cat([x, self._repeat(self.cls_emb, x.shape[0])], dim=1)
            cls_mask = self.build_cls_mask(text, cast_dtype)
            attn_mask = (
                attn_mask[None, :seq_len, :seq_len] + cls_mask[:, :seq_len, :seq_len]
            )

        x = x + self.positional_embedding[:seq_len].to(cast_dtype)
        # x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x, attn_mask=attn_mask, attn_method=attn_method)
        # x = x.permute(1, 0, 2)  # LND -> NLD

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        if self.cls_emb is not None:
            pooled, tokens = x[:, -1], x[:, :-1]
            pooled = self.ln_final(pooled)
        else:
            x = self.ln_final(x)
            pooled, tokens = x[torch.arange(x.shape[0]), text.argmax(dim=-1)], x

        if self.text_projection is not None:
            pooled = pooled @ self.text_projection

        self.hook.finalize()

        if self.output_tokens:
            return pooled, tokens

        return pooled
