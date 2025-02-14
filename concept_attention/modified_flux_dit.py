from dataclasses import dataclass

import torch
from torch import Tensor, nn

from concept_attention.flux.src.flux.modules.layers import (DoubleStreamBlock, EmbedND, LastLayer,
                                 MLPEmbedder, SingleStreamBlock,
                                 timestep_embedding)

from concept_attention.modified_double_stream_block import ModifiedDoubleStreamBlock
from concept_attention.modified_single_stream_block import ModifiedSingleStreamBlock

@dataclass
class FluxParams:
    in_channels: int
    vec_in_dim: int
    context_in_dim: int
    hidden_size: int
    mlp_ratio: float
    num_heads: int
    depth: int
    depth_single_blocks: int
    axes_dim: list[int]
    theta: int
    qkv_bias: bool
    guidance_embed: bool


class ModifiedFluxDiT(nn.Module):
    """
    Transformer model for flow matching on sequences.
    """

    def __init__(self, params: FluxParams, attention_block_class=ModifiedDoubleStreamBlock):
        super().__init__()

        self.params = params
        self.in_channels = params.in_channels
        self.out_channels = self.in_channels
        if params.hidden_size % params.num_heads != 0:
            raise ValueError(
                f"Hidden size {params.hidden_size} must be divisible by num_heads {params.num_heads}"
            )
        pe_dim = params.hidden_size // params.num_heads
        if sum(params.axes_dim) != pe_dim:
            raise ValueError(f"Got {params.axes_dim} but expected positional dim {pe_dim}")
        self.hidden_size = params.hidden_size
        self.num_heads = params.num_heads
        self.pe_embedder = EmbedND(dim=pe_dim, theta=params.theta, axes_dim=params.axes_dim)
        self.img_in = nn.Linear(self.in_channels, self.hidden_size, bias=True)
        self.time_in = MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size)
        self.vector_in = MLPEmbedder(params.vec_in_dim, self.hidden_size)
        self.guidance_in = (
            MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size) if params.guidance_embed else nn.Identity()
        )
        self.txt_in = nn.Linear(params.context_in_dim, self.hidden_size)

        self.double_blocks = nn.ModuleList([
            attention_block_class(
                self.hidden_size,
                self.num_heads,
                mlp_ratio=params.mlp_ratio,
                qkv_bias=params.qkv_bias,
            )
            for _ in range(params.depth)
        ])

        self.single_blocks = nn.ModuleList([
            ModifiedSingleStreamBlock(self.hidden_size, self.num_heads, mlp_ratio=params.mlp_ratio)
            for _ in range(params.depth_single_blocks)
        ])

        self.final_layer = LastLayer(self.hidden_size, 1, self.out_channels)

    def forward(
        self,
        img: Tensor,
        img_ids: Tensor,
        txt: Tensor,
        txt_ids: Tensor,
        concepts: Tensor,
        concept_ids: Tensor,
        concept_vec: Tensor,
        timesteps: Tensor,
        y: Tensor,
        guidance: Tensor | None = None,
        stop_after_multimodal_attentions: bool = False,
        edit_metadata=None,
        iteration=None,
        joint_attention_kwargs=None,
        **kwargs
    ) -> Tensor:
        assert concept_vec is not None, "Concept vectors must be provided for this implementation."
        if img.ndim != 3 or txt.ndim != 3:
            raise ValueError("Input img and txt tensors must have 3 dimensions.")

        # running on sequences img
        img = self.img_in(img)
        vec = self.time_in(timestep_embedding(timesteps, 256))
        if self.params.guidance_embed:
            if guidance is None:
                raise ValueError("Didn't get guidance strength for guidance distilled model.")
            vec = vec + self.guidance_in(timestep_embedding(guidance, 256))
        vec = vec + self.vector_in(y)
        txt = self.txt_in(txt)

        ids = torch.cat((txt_ids, img_ids), dim=1)
        pe = self.pe_embedder(ids)
        # Compute positional encodings
        ids_with_concepts = torch.cat((concept_ids, img_ids), dim=1)
        pe_with_concepts = self.pe_embedder(ids_with_concepts)
        ################ Process concept vectors ################
        original_concept_vec = concept_vec
        concept_vec = self.time_in(timestep_embedding(timesteps, 256))
        if self.params.guidance_embed:
            if guidance is None:
                raise ValueError("Didn't get guidance strength for guidance distilled model.")
            concept_vec = concept_vec + self.guidance_in(timestep_embedding(guidance, 256))
        concept_vec = concept_vec + self.vector_in(original_concept_vec)
        concepts = self.txt_in(concepts)
        ############## Modify the double blocks to also return concept vectors ##############
        all_cross_attention_maps = []
        all_concept_attention_maps = []
        for block in self.double_blocks:
            img, txt, concepts, cross_attention_maps, concept_attention_maps = block(
                img=img, 
                txt=txt, 
                vec=vec, 
                pe=pe,
                concepts=concepts,
                concept_vec=concept_vec,
                concept_pe=pe_with_concepts,
                edit_metadata=edit_metadata,
                iteration=iteration,
                joint_attention_kwargs=joint_attention_kwargs
            )
            all_cross_attention_maps.append(cross_attention_maps)
            all_concept_attention_maps.append(concept_attention_maps)   
        
        all_concept_attention_maps = torch.stack(all_concept_attention_maps, dim=0)
        all_cross_attention_maps = torch.stack(all_cross_attention_maps, dim=0)
        #####################################################################################

        img = torch.cat((txt, img), 1)
        
        # Speed up segmentation by not generating the full image
        if stop_after_multimodal_attentions:
            return None, all_cross_attention_maps, all_concept_attention_maps

        # Do the single blocks now
        for block in self.single_blocks:
            img = block(img, vec=vec, pe=pe)

        img = img[:, txt.shape[1] :, ...]

        img = self.final_layer(img, vec)  # (N, T, patch_size ** 2 * out_channels)
        return img, all_cross_attention_maps, all_concept_attention_maps
