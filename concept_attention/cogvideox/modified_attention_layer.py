
from typing import Any, Dict, Optional, Tuple, Union
import einops

import torch
from torch import nn
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import PeftAdapterMixin
from diffusers.utils import USE_PEFT_BACKEND, is_torch_version, logging, scale_lora_layers, unscale_lora_layers
from diffusers.utils.torch_utils import maybe_allow_in_graph
from diffusers.models.attention import Attention, FeedForward
from diffusers.models.attention_processor import AttentionProcessor, FusedCogVideoXAttnProcessor2_0
from diffusers.models.embeddings import CogVideoXPatchEmbed, TimestepEmbedding, Timesteps
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import AdaLayerNorm, CogVideoXLayerNormZero


class CustomCogVideoXAttnProcessor2_0:
    r"""
    Processor for implementing scaled dot-product attention for the CogVideoX model. It applies a rotary embedding on
    query and key vectors, but does not include spatial normalization.
    """

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("CogVideoXAttnProcessor requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        concept_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        text_seq_length = encoder_hidden_states.size(1)

        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        batch_size, sequence_length, _ = hidden_states.shape

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        # Apply k, q, v to concept hidden states
        concept_query = attn.to_q(concept_hidden_states)
        concept_key = attn.to_k(concept_hidden_states)
        concept_value = attn.to_v(concept_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # Reshape concept query, key, value
        concept_query = concept_query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        concept_key = concept_key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        concept_value = concept_value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
            concept_query = attn.norm_q(concept_query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)
            concept_key = attn.norm_k(concept_key)

        # Apply RoPE if needed
        if image_rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb

            query[:, :, text_seq_length:] = apply_rotary_emb(query[:, :, text_seq_length:], image_rotary_emb)
            if not attn.is_cross_attention:
                key[:, :, text_seq_length:] = apply_rotary_emb(key[:, :, text_seq_length:], image_rotary_emb)

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        ###################### Concept Attention Projections ###########################
        # Package together the concept and image embeddings for attention
        image_query = query[:, :, text_seq_length:]
        image_key = key[:, :, text_seq_length:]
        image_value = value[:, :, text_seq_length:]
        concept_image_queries = torch.cat([concept_query, image_query], dim=2)
        concept_image_keys = torch.cat([concept_key, image_key], dim=2)
        concept_image_values = torch.cat([concept_value, image_value], dim=2)
        # Apply the attention to the concept and image embeddings
        concept_attn_hidden_states = F.scaled_dot_product_attention(
            concept_image_queries, 
            concept_image_keys, 
            concept_image_values, 
            # attn_mask=attention_mask, 
            dropout_p=0.0
        )
        # Pull out just the concept embedding outputs
        attn_concept_hidden_states = concept_attn_hidden_states[:, :, :concept_hidden_states.size(1)]
        ############# Compute Cross Attention Maps ##############
        cross_attention_maps = einops.einsum(
            image_query,
            concept_key,
            "batch heads patches dim, batch heads concepts dim -> batch heads concepts patches"
        )
        # Average over the heads
        cross_attention_maps = einops.reduce(
            cross_attention_maps,
            "batch heads concepts patches -> batch concepts patches",
            "mean"
        )
        ################################################################################
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        concept_hidden_states = attn_concept_hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        # ########## Concept Attention Projection ##########
        # Pull out the hidden states for the image patches 
        image_hidden_states = hidden_states[:, text_seq_length:]
        # Do the concept attention projections
        concept_attention_maps = einops.einsum(
            concept_hidden_states,
            image_hidden_states,
            "batch concepts dim, batch patches dim -> batch concepts patches"
        )
        # Save the info
        concept_attention_dict = {
            "concept_attention_maps": concept_attention_maps,
            "cross_attention_maps": cross_attention_maps,
        }
        # #############################################

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        concept_hidden_states = attn.to_out[0](concept_hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)
        concept_hidden_states = attn.to_out[1](concept_hidden_states)
        # Apply 

        encoder_hidden_states, hidden_states = hidden_states.split(
            [text_seq_length, hidden_states.size(1) - text_seq_length], dim=1
        )
        return hidden_states, encoder_hidden_states, concept_hidden_states, concept_attention_dict


# @maybe_allow_in_graph
class ModifiedCogVideoXBlock(nn.Module):
    r"""
    Transformer block used in [CogVideoX](https://github.com/THUDM/CogVideo) model.

    Parameters:
        dim (`int`):
            The number of channels in the input and output.
        num_attention_heads (`int`):
            The number of heads to use for multi-head attention.
        attention_head_dim (`int`):
            The number of channels in each head.
        time_embed_dim (`int`):
            The number of channels in timestep embedding.
        dropout (`float`, defaults to `0.0`):
            The dropout probability to use.
        activation_fn (`str`, defaults to `"gelu-approximate"`):
            Activation function to be used in feed-forward.
        attention_bias (`bool`, defaults to `False`):
            Whether or not to use bias in attention projection layers.
        qk_norm (`bool`, defaults to `True`):
            Whether or not to use normalization after query and key projections in Attention.
        norm_elementwise_affine (`bool`, defaults to `True`):
            Whether to use learnable elementwise affine parameters for normalization.
        norm_eps (`float`, defaults to `1e-5`):
            Epsilon value for normalization layers.
        final_dropout (`bool` defaults to `False`):
            Whether to apply a final dropout after the last feed-forward layer.
        ff_inner_dim (`int`, *optional*, defaults to `None`):
            Custom hidden dimension of Feed-forward layer. If not provided, `4 * dim` is used.
        ff_bias (`bool`, defaults to `True`):
            Whether or not to use bias in Feed-forward layer.
        attention_out_bias (`bool`, defaults to `True`):
            Whether or not to use bias in Attention output projection layer.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        time_embed_dim: int,
        dropout: float = 0.0,
        activation_fn: str = "gelu-approximate",
        attention_bias: bool = False,
        qk_norm: bool = True,
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        final_dropout: bool = True,
        ff_inner_dim: Optional[int] = None,
        ff_bias: bool = True,
        attention_out_bias: bool = True,
    ):
        super().__init__()

        # 1. Self Attention
        self.norm1 = CogVideoXLayerNormZero(time_embed_dim, dim, norm_elementwise_affine, norm_eps, bias=True)

        self.attn1 = Attention(
            query_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            qk_norm="layer_norm" if qk_norm else None,
            eps=1e-6,
            bias=attention_bias,
            out_bias=attention_out_bias,
            processor=CustomCogVideoXAttnProcessor2_0(),
        )

        # 2. Feed Forward
        self.norm2 = CogVideoXLayerNormZero(time_embed_dim, dim, norm_elementwise_affine, norm_eps, bias=True)

        self.ff = FeedForward(
            dim,
            dropout=dropout,
            activation_fn=activation_fn,
            final_dropout=final_dropout,
            inner_dim=ff_inner_dim,
            bias=ff_bias,
        )

    @torch.no_grad()
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        concept_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> torch.Tensor:
        text_seq_length = encoder_hidden_states.size(1)

        # norm & modulate
        norm_hidden_states, norm_encoder_hidden_states, gate_msa, enc_gate_msa = self.norm1(
            hidden_states, encoder_hidden_states, temb
        )

        ################## Concept Attention ##################

        _, norm_concept_hidden_states, _, concept_gate_msa = self.norm1(
            hidden_states, concept_hidden_states, temb
        )

        # Concept attention
        attn_hidden_states, attn_encoder_hidden_states, attn_concept_hidden_states, concept_attention_dict = self.attn1(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            concept_hidden_states=norm_concept_hidden_states,
            image_rotary_emb=image_rotary_emb
        )

        concept_hidden_states = concept_hidden_states + concept_gate_msa * attn_concept_hidden_states

        _, norm_concept_hidden_states, _, concept_gate_ff = self.norm2(
            hidden_states, concept_hidden_states, temb
        )

        concept_ff_output = self.ff(norm_concept_hidden_states)

        concept_hidden_states = concept_hidden_states + concept_gate_ff * concept_ff_output

        ######################################################

        # Now do the normal attention

        hidden_states = hidden_states + gate_msa * attn_hidden_states
        encoder_hidden_states = encoder_hidden_states + enc_gate_msa * attn_encoder_hidden_states

        # norm & modulate
        norm_hidden_states, norm_encoder_hidden_states, gate_ff, enc_gate_ff = self.norm2(
            hidden_states, encoder_hidden_states, temb
        )
     
        # feed-forward
        norm_hidden_states = torch.cat([norm_encoder_hidden_states, norm_hidden_states], dim=1)
        ff_output = self.ff(norm_hidden_states)

       
        hidden_states = hidden_states + gate_ff * ff_output[:, text_seq_length:]
        encoder_hidden_states = encoder_hidden_states + enc_gate_ff * ff_output[:, :text_seq_length]

        return hidden_states, encoder_hidden_states, concept_hidden_states, concept_attention_dict
