# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch import Tensor
# from einops import rearrange
# import einops

# from flux2.src.flux2.model import SelfAttention, attention
# from flux2.src.flux2.model import SiLUActivation


# class ModifiedDoubleStreamBlock(nn.Module):
#     def __init__(
#         self,
#         hidden_size: int,
#         num_heads: int,
#         mlp_ratio: float,
#     ):
#         super().__init__()
#         mlp_hidden_dim = int(hidden_size * mlp_ratio)
#         self.num_heads = num_heads
#         assert (
#             hidden_size % num_heads == 0
#         ), f"{hidden_size=} must be divisible by {num_heads=}"

#         self.hidden_size = hidden_size
#         self.img_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
#         self.mlp_mult_factor = 2

#         self.img_attn = SelfAttention(
#             dim=hidden_size,
#             num_heads=num_heads,
#         )

#         self.img_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
#         self.img_mlp = nn.Sequential(
#             nn.Linear(hidden_size, mlp_hidden_dim * self.mlp_mult_factor, bias=False),
#             SiLUActivation(),
#             nn.Linear(mlp_hidden_dim, hidden_size, bias=False),
#         )

#         self.txt_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
#         self.txt_attn = SelfAttention(
#             dim=hidden_size,
#             num_heads=num_heads,
#         )

#         self.txt_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
#         self.txt_mlp = nn.Sequential(
#             nn.Linear(
#                 hidden_size,
#                 mlp_hidden_dim * self.mlp_mult_factor,
#                 bias=False,
#             ),
#             SiLUActivation(),
#             nn.Linear(mlp_hidden_dim, hidden_size, bias=False),
#         )

#     def forward(
#         self,
#         img: Tensor,
#         txt: Tensor,
#         concepts: Tensor,  # Adding concept tensor input
#         pe_img: Tensor,
#         pe_ctx: Tensor,
#         pe_concept: Tensor,  # Adding concept positional encodings
#         mod_img: tuple[Tensor, Tensor],
#         mod_txt: tuple[Tensor, Tensor],
#     ) -> tuple[Tensor, Tensor]:
#         img_mod1, img_mod2 = mod_img
#         txt_mod1, txt_mod2 = mod_txt
#         # Prepare image for attention
#         img_mod1_shift, img_mod1_scale, img_mod1_gate = img_mod1
#         img_mod2_shift, img_mod2_scale, img_mod2_gate = img_mod2
#         txt_mod1_shift, txt_mod1_scale, txt_mod1_gate = txt_mod1
#         txt_mod2_shift, txt_mod2_scale, txt_mod2_gate = txt_mod2

#         # prepare image for attention
#         img_modulated = self.img_norm1(img)
#         img_modulated = (1 + img_mod1_scale) * img_modulated + img_mod1_shift

#         img_qkv = self.img_attn.qkv(img_modulated)
#         img_q, img_k, img_v = rearrange(
#             img_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads
#         )
#         img_q, img_k = self.img_attn.norm(img_q, img_k, img_v)

#         # prepare txt for attention
#         txt_modulated = self.txt_norm1(txt)
#         txt_modulated = (1 + txt_mod1_scale) * txt_modulated + txt_mod1_shift

#         txt_qkv = self.txt_attn.qkv(txt_modulated)
#         txt_q, txt_k, txt_v = rearrange(
#             txt_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads
#         )
#         txt_q, txt_k = self.txt_attn.norm(txt_q, txt_k, txt_v)

#         # ################### Prepare concept for attention ###################
#         # # Use text modulation on concept tokens
#         # concept_modulated = self.txt_norm1(concepts)
#         # concept_modulated = (1 + txt_mod1_scale) * concept_modulated + txt_mod1_shift
#         # # Create the qkv for concept tokens using txt_attn
#         # concept_qkv = self.txt_attn.qkv(concept_modulated)
#         # concept_q, concept_k, concept_v = rearrange(
#         #     concept_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads
#         # )
#         # concept_q, concept_k = self.txt_attn.norm(concept_q, concept_k, concept_v)
#         # #####################################################################

#         q = torch.cat((txt_q, img_q), dim=2)
#         k = torch.cat((txt_k, img_q), dim=2)
#         v = torch.cat((txt_v, img_v), dim=2)

#         pe_ctx_img = torch.cat((pe_ctx, pe_img), dim=2)
#         attn = attention(q, k, v, pe_ctx_img)
#         txt_attn = attn[:, : txt_q.shape[2]]
#         img_attn = attn[:, txt_q.shape[2] :]

#         # ################### Do an attention operation with the concepts ###################
#         # q = torch.cat((concept_q, img_q), dim=2)
#         # k = torch.cat((concept_k, img_q), dim=2)
#         # v = torch.cat((concept_v, img_v), dim=2)
#         # pe_concepts_image = torch.cat((pe_concept, pe_img), dim=2)
#         # concept_img_attn = attention(q, k, v, pe_concepts_image)
#         # concept_attn = concept_img_attn[:, : concept_q.shape[2]]
#         # ###########################################################################
#         # ############# Cache the concept output space vectors here #############
#         # concept_attention_dict = {}
#         # concept_attention_dict["concept_attn"] = concept_attn.detach()
#         # # Also cache the image output space vectors for analysis
#         # concept_attention_dict["img_attn"] = img_attn.detach()
#         # #######################################################################

#         # calculate the img blocks
#         img = img + img_mod1_gate * self.img_attn.proj(img_attn)
#         img = img + img_mod2_gate * self.img_mlp(
#             (1 + img_mod2_scale) * (self.img_norm2(img)) + img_mod2_shift
#         )

#         # calculate the txt blocks
#         txt = txt + txt_mod1_gate * self.txt_attn.proj(txt_attn)
#         txt = txt + txt_mod2_gate * self.txt_mlp(
#             (1 + txt_mod2_scale) * (self.txt_norm2(txt)) + txt_mod2_shift
#         )

#         # ################# Do the projection for concepts as well #################
#         # concepts = concepts + txt_mod1_gate * self.txt_attn.proj(concept_attn)
#         # concepts = concepts + txt_mod2_gate * self.txt_mlp(
#         #     (1 + txt_mod2_scale) * (self.txt_norm2(concepts)) + txt_mod2_shift
#         # )
#         # ########################################################################

#         return img, txt, concepts, {}  # concept_attention_dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from einops import rearrange
import einops

from flux2.src.flux2.model import SelfAttention, attention, SiLUActivation


# class DoubleStreamBlock(nn.Module):
#     def __init__(
#         self,
#         hidden_size: int,
#         num_heads: int,
#         mlp_ratio: float,
#     ):
#         super().__init__()
#         mlp_hidden_dim = int(hidden_size * mlp_ratio)
#         self.num_heads = num_heads
#         assert (
#             hidden_size % num_heads == 0
#         ), f"{hidden_size=} must be divisible by {num_heads=}"

#         self.hidden_size = hidden_size
#         self.img_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
#         self.mlp_mult_factor = 2

#         self.img_attn = SelfAttention(
#             dim=hidden_size,
#             num_heads=num_heads,
#         )

#         self.img_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
#         self.img_mlp = nn.Sequential(
#             nn.Linear(hidden_size, mlp_hidden_dim * self.mlp_mult_factor, bias=False),
#             SiLUActivation(),
#             nn.Linear(mlp_hidden_dim, hidden_size, bias=False),
#         )

#         self.txt_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
#         self.txt_attn = SelfAttention(
#             dim=hidden_size,
#             num_heads=num_heads,
#         )

#         self.txt_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
#         self.txt_mlp = nn.Sequential(
#             nn.Linear(
#                 hidden_size,
#                 mlp_hidden_dim * self.mlp_mult_factor,
#                 bias=False,
#             ),
#             SiLUActivation(),
#             nn.Linear(mlp_hidden_dim, hidden_size, bias=False),
#         )

#     def forward(
#         self,
#         img: Tensor,
#         txt: Tensor,
#         # concepts: Tensor,  # Adding concept tensor input
#         pe_img: Tensor,
#         pe_ctx: Tensor,
#         # pe_concept: Tensor,  # Adding concept positional encodings
#         mod_img: tuple[Tensor, Tensor],
#         mod_txt: tuple[Tensor, Tensor],
#     ) -> tuple[Tensor, Tensor]:
#         img_mod1, img_mod2 = mod_img
#         txt_mod1, txt_mod2 = mod_txt
#         # Prepare image for attention
#         img_mod1_shift, img_mod1_scale, img_mod1_gate = img_mod1
#         img_mod2_shift, img_mod2_scale, img_mod2_gate = img_mod2
#         txt_mod1_shift, txt_mod1_scale, txt_mod1_gate = txt_mod1
#         txt_mod2_shift, txt_mod2_scale, txt_mod2_gate = txt_mod2

#         # prepare image for attention
#         img_modulated = self.img_norm1(img)
#         img_modulated = (1 + img_mod1_scale) * img_modulated + img_mod1_shift

#         img_qkv = self.img_attn.qkv(img_modulated)
#         img_q, img_k, img_v = rearrange(
#             img_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads
#         )
#         img_q, img_k = self.img_attn.norm(img_q, img_k, img_v)

#         # prepare txt for attention
#         txt_modulated = self.txt_norm1(txt)
#         txt_modulated = (1 + txt_mod1_scale) * txt_modulated + txt_mod1_shift

#         txt_qkv = self.txt_attn.qkv(txt_modulated)
#         txt_q, txt_k, txt_v = rearrange(
#             txt_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads
#         )
#         txt_q, txt_k = self.txt_attn.norm(txt_q, txt_k, txt_v)

#         q = torch.cat((txt_q, img_q), dim=2)
#         k = torch.cat((txt_k, img_q), dim=2)
#         v = torch.cat((txt_v, img_v), dim=2)

#         pe_ctx_img = torch.cat((pe_ctx, pe_img), dim=2)
#         attn = attention(q, k, v, pe_ctx_img)
#         txt_attn = attn[:, : txt_q.shape[2]]
#         img_attn = attn[:, txt_q.shape[2] :]

#         # calculate the img blocks
#         img = img + img_mod1_gate * self.img_attn.proj(img_attn)
#         img = img + img_mod2_gate * self.img_mlp(
#             (1 + img_mod2_scale) * (self.img_norm2(img)) + img_mod2_shift
#         )

#         # calculate the txt blocks
#         txt = txt + txt_mod1_gate * self.txt_attn.proj(txt_attn)
#         txt = txt + txt_mod2_gate * self.txt_mlp(
#             (1 + txt_mod2_scale) * (self.txt_norm2(txt)) + txt_mod2_shift
#         )

#         return img, txt, None, {}  # concept_attention_dict


class ModifiedDoubleStreamBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float,
    ):
        super().__init__()
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.num_heads = num_heads
        assert (
            hidden_size % num_heads == 0
        ), f"{hidden_size=} must be divisible by {num_heads=}"

        self.hidden_size = hidden_size
        self.img_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp_mult_factor = 2

        self.img_attn = SelfAttention(
            dim=hidden_size,
            num_heads=num_heads,
        )

        self.img_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim * self.mlp_mult_factor, bias=False),
            SiLUActivation(),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=False),
        )

        self.txt_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_attn = SelfAttention(
            dim=hidden_size,
            num_heads=num_heads,
        )

        self.txt_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_mlp = nn.Sequential(
            nn.Linear(
                hidden_size,
                mlp_hidden_dim * self.mlp_mult_factor,
                bias=False,
            ),
            SiLUActivation(),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=False),
        )

    def forward(
        self,
        img: Tensor,
        txt: Tensor,
        concepts: Tensor,  # Adding concept tensor input
        pe_image: Tensor,
        pe_ctx: Tensor,
        pe_concept: Tensor,
        mod_img: tuple[Tensor, Tensor],
        mod_txt: tuple[Tensor, Tensor],
    ) -> tuple[Tensor, Tensor]:
        img_mod1, img_mod2 = mod_img
        txt_mod1, txt_mod2 = mod_txt

        img_mod1_shift, img_mod1_scale, img_mod1_gate = img_mod1
        img_mod2_shift, img_mod2_scale, img_mod2_gate = img_mod2
        txt_mod1_shift, txt_mod1_scale, txt_mod1_gate = txt_mod1
        txt_mod2_shift, txt_mod2_scale, txt_mod2_gate = txt_mod2

        # prepare image for attention
        img_modulated = self.img_norm1(img)
        img_modulated = (1 + img_mod1_scale) * img_modulated + img_mod1_shift

        img_qkv = self.img_attn.qkv(img_modulated)
        img_q, img_k, img_v = rearrange(
            img_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads
        )
        img_q, img_k = self.img_attn.norm(img_q, img_k, img_v)

        # prepare txt for attention
        txt_modulated = self.txt_norm1(txt)
        txt_modulated = (1 + txt_mod1_scale) * txt_modulated + txt_mod1_shift

        txt_qkv = self.txt_attn.qkv(txt_modulated)
        txt_q, txt_k, txt_v = rearrange(
            txt_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads
        )
        txt_q, txt_k = self.txt_attn.norm(txt_q, txt_k, txt_v)

        ################### Prepare concept for attention ###################
        # Use text modulation on concept tokens
        concept_modulated = self.txt_norm1(concepts)
        concept_modulated = (1 + txt_mod1_scale) * concept_modulated + txt_mod1_shift
        # Create the qkv for concept tokens using txt_attn
        concept_qkv = self.txt_attn.qkv(concept_modulated)
        concept_q, concept_k, concept_v = rearrange(
            concept_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads
        )
        concept_q, concept_k = self.txt_attn.norm(concept_q, concept_k, concept_v)
        #####################################################################

        q = torch.cat((txt_q, img_q), dim=2)
        k = torch.cat((txt_k, img_k), dim=2)
        v = torch.cat((txt_v, img_v), dim=2)

        pe = torch.cat((pe_ctx, pe_image), dim=2)
        attn = attention(q, k, v, pe)
        txt_attn, img_attn = attn[:, : txt_q.shape[2]], attn[:, txt_q.shape[2] :]

        ################### Do an attention operation with the concepts ###################

        q = torch.cat((concept_q, img_q), dim=2)
        k = torch.cat((concept_k, img_k), dim=2)
        v = torch.cat((concept_v, img_v), dim=2)
        pe_concepts_image = torch.cat((pe_concept, pe_image), dim=2)
        concept_img_attn = attention(q, k, v, pe_concepts_image)
        concept_attn = concept_img_attn[:, : concept_q.shape[2]]
        ###########################################################################
        ############# Save the concept and image ouptut vectors for analysis #############
        concept_attention_dict = {}
        # Concept attention outputs: (batch_size, num_concepts, dim)
        # Image attention outputs: (batch_size, num_image_tokens, dim)
        concept_outputs = concept_attn.detach()
        img_outputs = img_attn.detach()
        concept_scores = einops.einsum(
            concept_outputs,
            img_outputs,
            "batch num_concepts dim, batch num_img_tokens dim -> batch num_concepts num_img_tokens",
        )
        concept_attention_dict["concept_scores"] = concept_scores.cpu()
        ##################################################################

        # calculate the img blocks
        img = img + img_mod1_gate * self.img_attn.proj(img_attn)
        img = img + img_mod2_gate * self.img_mlp(
            (1 + img_mod2_scale) * (self.img_norm2(img)) + img_mod2_shift
        )

        # calculate the txt blocks
        txt = txt + txt_mod1_gate * self.txt_attn.proj(txt_attn)
        txt = txt + txt_mod2_gate * self.txt_mlp(
            (1 + txt_mod2_scale) * (self.txt_norm2(txt)) + txt_mod2_shift
        )

        ################### Do the projection for concepts as well #################
        concepts = concepts + txt_mod1_gate * self.txt_attn.proj(concept_attn)
        concepts = concepts + txt_mod2_gate * self.txt_mlp(
            (1 + txt_mod2_scale) * (self.txt_norm2(concepts)) + txt_mod2_shift
        )
        ########################################################################

        return img, txt, concepts, concept_attention_dict
