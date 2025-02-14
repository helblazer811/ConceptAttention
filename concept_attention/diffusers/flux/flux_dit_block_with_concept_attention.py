
import torch
from typing import Any, Dict, Optional, Tuple
from torch import nn
import einops

concept_attention_default_kwargs = {
    "concept_attention_layers": list(range(10, 18)),
    "concepts": None,
}

from diffusers.models.transformers.transformer_flux import FluxTransformerBlock

class FluxTransformerBlockWithConceptAttention(FluxTransformerBlock):
    r"""
    A Transformer block following the MMDiT architecture, introduced in Stable Diffusion 3 with Concept Attention.
    """

    @torch.no_grad()
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        concept_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        concept_temb: torch.Tensor,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        concept_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        concept_attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
            hidden_states, 
            emb=temb
        )
        norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(
            encoder_hidden_states, 
            emb=temb
        )
        joint_attention_kwargs = joint_attention_kwargs or {}
        # Attention 
        attention_outputs = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
            **joint_attention_kwargs,
        )

        if len(attention_outputs) == 2:
            attn_output, context_attn_output = attention_outputs
        elif len(attention_outputs) == 3:
            attn_output, context_attn_output, ip_attn_output = attention_outputs
        ################################ Do Concept Attention ################################
        if concept_attention_kwargs is not None and concept_hidden_states is not None:
            # Normalize the concept hidden states
            norm_concept_hidden_states, concept_gate_msa, concept_shift_mlp, concept_scale_mlp, concept_gate_mlp = self.norm1_context(
                concept_hidden_states, 
                emb=concept_temb
            )
            # Process the attention outputs for the concept_hidden_states.
            # NOTE: This does some unecessary computations, but it is fine for now.
            concept_attention_outputs = self.attn(
                hidden_states=norm_hidden_states,
                encoder_hidden_states=norm_concept_hidden_states,
                image_rotary_emb=concept_rotary_emb,
                **joint_attention_kwargs,
            )
            # Unpack the attention outputs
            if len(attention_outputs) == 2:
                _, concept_attn_output = concept_attention_outputs
            elif len(attention_outputs) == 3:
                _, concept_attn_output, _ = concept_attention_outputs
            # Now compute the concept attention maps
            concept_attention_map = einops.einsum(
                concept_attn_output, # Concept attention output
                attn_output, # Image attention output
                "batch concepts dim, batch patches dim -> batch concepts patches", # Einsum equation
            )
            # Detach and move to cpu the concept attention map
            concept_attention_map = concept_attention_map.detach().cpu()
            # Now do the residual stream update 
            concept_attn_output = concept_gate_msa.unsqueeze(1) * concept_attn_output
            concept_hidden_states = concept_hidden_states + concept_attn_output
            norm_concept_hidden_states = self.norm2_context(concept_hidden_states)
            norm_concept_hidden_states = norm_concept_hidden_states * (1 + concept_scale_mlp[:, None]) + concept_shift_mlp[:, None]
            concept_ff_output = self.ff_context(norm_concept_hidden_states)
            concept_hidden_states = concept_hidden_states + concept_gate_mlp.unsqueeze(1) * concept_ff_output
            if concept_hidden_states.dtype == torch.float16:
                concept_hidden_states = concept_hidden_states.clip(-65504, 65504)
        else:
            concept_attention_map = None
            concept_hidden_states = None
        ######################################################################################

        # Process attention outputs for the `hidden_states`.
        attn_output = gate_msa.unsqueeze(1) * attn_output
        hidden_states = hidden_states + attn_output

        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

        ff_output = self.ff(norm_hidden_states)
        ff_output = gate_mlp.unsqueeze(1) * ff_output

        hidden_states = hidden_states + ff_output
        if len(attention_outputs) == 3:
            hidden_states = hidden_states + ip_attn_output

        # Process attention outputs for the `encoder_hidden_states`.

        context_attn_output = c_gate_msa.unsqueeze(1) * context_attn_output
        encoder_hidden_states = encoder_hidden_states + context_attn_output

        norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
        norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]

        context_ff_output = self.ff_context(norm_encoder_hidden_states)
        encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output
        if encoder_hidden_states.dtype == torch.float16:
            encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)

        return encoder_hidden_states, hidden_states, concept_hidden_states, concept_attention_map