import torch
from torch import nn, Tensor
import einops
import math
import torch.nn.functional as F
import matplotlib.pyplot as plt

from concept_attention.flux.src.flux.modules.layers import Modulation, SelfAttention
from concept_attention.flux.src.flux.math import apply_rope


def attention(q: Tensor, k: Tensor, v: Tensor, pe: Tensor) -> Tensor:
    q, k = apply_rope(q, k, pe)

    x = scaled_dot_product_attention(q, k, v)
    x = einops.rearrange(x, "B H L D -> B L (H D)")

    return x

# Efficient implementation equivalent to the following:
def scaled_dot_product_attention(
    query, 
    key, 
    value,
    attn_mask=None
) -> torch.Tensor:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1))
    attn_bias = torch.zeros(L, S, dtype=query.dtype).to(query.device)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias += attn_mask

    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)

    return attn_weight @ value

class ModifiedDoubleStreamBlock(nn.Module):

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float, qkv_bias: bool = False):
        super().__init__()
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.img_mod = Modulation(hidden_size, double=True)
        self.img_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_attn = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias)
        self.img_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )
        self.txt_mod = Modulation(hidden_size, double=True)
        self.txt_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_attn = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias)
        self.txt_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )

    @torch.no_grad()
    def forward(
        self, 
        img: Tensor, 
        txt: Tensor, 
        vec: Tensor, 
        pe: Tensor, 
        concepts: Tensor, 
        concept_vec: Tensor,
        concept_pe: Tensor,
        joint_attention_kwargs=None,
        **kwargs
    ) -> tuple[Tensor, Tensor]:
        assert concept_vec is not None, "Concept vectors must be provided for this implementation."
        img_mod1, img_mod2 = self.img_mod(vec)
        txt_mod1, txt_mod2 = self.txt_mod(vec)
        concept_mod1, concept_mod2 = self.txt_mod(concept_vec)
        # Prepare image for attention
        img_modulated = self.img_norm1(img)
        img_modulated = (1 + img_mod1.scale) * img_modulated + img_mod1.shift
        img_qkv = self.img_attn.qkv(img_modulated)
        img_q, img_k, img_v = einops.rearrange(img_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        img_q, img_k = self.img_attn.norm(img_q, img_k, img_v)
        # Prepare txt for attention
        txt_modulated = self.txt_norm1(txt)
        txt_modulated = (1 + txt_mod1.scale) * txt_modulated + txt_mod1.shift
        txt_qkv = self.txt_attn.qkv(txt_modulated)
        txt_q, txt_k, txt_v = einops.rearrange(txt_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        txt_q, txt_k = self.txt_attn.norm(txt_q, txt_k, txt_v)
        # Prepare concepts for attention
        concept_modulated = self.txt_norm1(concepts)
        concept_modulated = (1 + concept_mod1.scale) * concept_modulated + concept_mod1.shift
        concept_qkv = self.txt_attn.qkv(concept_modulated)
        concept_q, concept_k, concept_v = einops.rearrange(concept_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        concept_q, concept_k = self.txt_attn.norm(concept_q, concept_k, concept_v)
        ########## Do the text-image joint attention ##########
        text_image_q = torch.cat((txt_q, img_q), dim=2)
        text_image_k = torch.cat((txt_k, img_k), dim=2)
        text_image_v = torch.cat((txt_v, img_v), dim=2)
        # Apply rope
        text_image_q, text_image_k = apply_rope(text_image_q, text_image_k, pe)
        # Do the attention operation
        text_image_attn = F.scaled_dot_product_attention(
            text_image_q, 
            text_image_k,
            text_image_v
        )
        # Separate the text and image attentions
        txt_attn = text_image_attn[:, :, :txt.shape[1]]
        img_attn = text_image_attn[:, :, txt.shape[1]:]
        ########## Do the concept-image joint attention ##########
        concept_image_q = torch.cat((concept_q, img_q), dim=2)
        concept_image_k = torch.cat((concept_k, img_k), dim=2)
        concept_image_v = torch.cat((concept_v, img_v), dim=2)
        # Apply rope
        concept_image_q, concept_image_k = apply_rope(concept_image_q, concept_image_k, concept_pe)
        if joint_attention_kwargs is not None:
            concept_cross_attention = joint_attention_kwargs.get("concept_cross_attention", True)
            concept_self_attention = joint_attention_kwargs.get("concept_self_attention", True)
            if concept_cross_attention and not concept_self_attention:
                # Do cross attention only between concepts and image
                concept_only_q = concept_image_q[:, :, :concepts.shape[1]]
                image_only_k = concept_image_k[:, :, concepts.shape[1]:]
                # Do the attention operation
                concept_attn = scaled_dot_product_attention(
                    concept_only_q,
                    image_only_k,
                    img_v
                )
            elif concept_self_attention and not concept_cross_attention:
                concept_q = concept_image_q[:, :, :concepts.shape[1]]
                concept_k = concept_image_k[:, :, :concepts.shape[1]]
                # Do the attention operation
                concept_attn = scaled_dot_product_attention(
                    concept_q,
                    concept_k,
                    concept_v
                )
            elif concept_cross_attention and concept_self_attention: 
                # Do the attention operation
                concept_image_attn = F.scaled_dot_product_attention(
                    concept_image_q, 
                    concept_image_k, 
                    concept_image_v,
                )
                # Separate the concept and image attentions
                concept_attn = concept_image_attn[:, :, :concepts.shape[1]]
            else:
                # Neither self or cross. 
                concept_attn = concept_v
        else:
            # Do both cross and self attention
            concept_image_attn = F.scaled_dot_product_attention(
                concept_image_q, 
                concept_image_k, 
                concept_image_v,
            )
            # Separate the concept and image attentions
            concept_attn = concept_image_attn[:, :, :concepts.shape[1]]

        # Rearrange the attention tensors
        txt_attn = einops.rearrange(txt_attn, "B H L D -> B L (H D)")
        if joint_attention_kwargs is not None and joint_attention_kwargs.get("keep_head_dim", False):
            concept_attn = einops.rearrange(concept_attn, "B H L D -> B L (H D)")
            img_attn = einops.rearrange(img_attn, "B H L D -> B L (H D)")
        else:
            concept_attn = einops.rearrange(concept_attn, "B H L D -> B L (H D)")
            img_attn = einops.rearrange(img_attn, "B H L D -> B L (H D)")

        # Compute the cross attentions
        cross_attention_maps = einops.einsum(
            concept_q,
            img_q,
            "batch head concepts dim, batch had patches dim -> batch head concepts patches"
        )
        cross_attention_maps = einops.reduce(cross_attention_maps, "batch head concepts patches -> batch concepts patches", reduction="mean")
        # Compute the concept attentions
        concept_attention_maps = einops.einsum(
            concept_attn,
            img_attn,
            "batch concepts dim, batch patches dim -> batch concepts patches"
        )
        # Do the block updates
        # Calculate the img blocks
        img = img + img_mod1.gate * self.img_attn.proj(img_attn)
        # Can I do the decomposition here? Using a basis formed by (img_mod1.gate * self.img_attn.proj(concepts))
        img = img + img_mod2.gate * self.img_mlp((1 + img_mod2.scale) * self.img_norm2(img) + img_mod2.shift)
        # Calculate the txt blocks
        txt = txt + txt_mod1.gate * self.txt_attn.proj(txt_attn)
        txt = txt + txt_mod2.gate * self.txt_mlp((1 + txt_mod2.scale) * self.txt_norm2(txt) + txt_mod2.shift)
        # Calculate the concept blocks
        concepts = concepts + concept_mod1.gate * self.txt_attn.proj(concept_attn)
        concepts = concepts + concept_mod2.gate * self.txt_mlp((1 + concept_mod2.scale) * self.txt_norm2(concepts) + concept_mod2.shift)

        return img, txt, concepts, cross_attention_maps, concept_attention_maps