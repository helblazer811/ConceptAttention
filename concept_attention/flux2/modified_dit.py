import torch
import torch.nn as nn
from torch import Tensor

from flux2.src.flux2.model import (
    DoubleStreamBlock,
    EmbedND,
    Flux2Params,
    LastLayer,
    MLPEmbedder,
    Modulation,
    SingleStreamBlock,
    timestep_embedding,
)

from modified_double_stream import ModifiedDoubleStreamBlock


# class ModifiedFlux2(nn.Module):
#     def __init__(self, params: Flux2Params):
#         super().__init__()

#         self.in_channels = params.in_channels
#         self.out_channels = params.in_channels
#         if params.hidden_size % params.num_heads != 0:
#             raise ValueError(
#                 f"Hidden size {params.hidden_size} must be divisible by num_heads {params.num_heads}"
#             )
#         pe_dim = params.hidden_size // params.num_heads
#         if sum(params.axes_dim) != pe_dim:
#             raise ValueError(
#                 f"Got {params.axes_dim} but expected positional dim {pe_dim}"
#             )
#         self.hidden_size = params.hidden_size
#         self.num_heads = params.num_heads
#         self.pe_embedder = EmbedND(
#             dim=pe_dim, theta=params.theta, axes_dim=params.axes_dim
#         )
#         self.img_in = nn.Linear(self.in_channels, self.hidden_size, bias=False)
#         self.time_in = MLPEmbedder(
#             in_dim=256, hidden_dim=self.hidden_size, disable_bias=True
#         )
#         self.guidance_in = MLPEmbedder(
#             in_dim=256, hidden_dim=self.hidden_size, disable_bias=True
#         )
#         self.txt_in = nn.Linear(params.context_in_dim, self.hidden_size, bias=False)

#         self.double_blocks = nn.ModuleList(
#             [
#                 ModifiedDoubleStreamBlock(
#                     self.hidden_size,
#                     self.num_heads,
#                     mlp_ratio=params.mlp_ratio,
#                 )
#                 for _ in range(params.depth)
#             ]
#         )

#         self.single_blocks = nn.ModuleList(
#             [
#                 SingleStreamBlock(
#                     self.hidden_size,
#                     self.num_heads,
#                     mlp_ratio=params.mlp_ratio,
#                 )
#                 for _ in range(params.depth_single_blocks)
#             ]
#         )

#         self.double_stream_modulation_img = Modulation(
#             self.hidden_size,
#             double=True,
#             disable_bias=True,
#         )
#         self.double_stream_modulation_txt = Modulation(
#             self.hidden_size,
#             double=True,
#             disable_bias=True,
#         )
#         self.single_stream_modulation = Modulation(
#             self.hidden_size, double=False, disable_bias=True
#         )

#         self.final_layer = LastLayer(
#             self.hidden_size,
#             self.out_channels,
#         )

#     def forward(
#         self,
#         x: Tensor,
#         x_ids: Tensor,
#         timesteps: Tensor,
#         ctx: Tensor,
#         ctx_ids: Tensor,
#         concepts: Tensor,
#         concept_ids: Tensor,
#         guidance: Tensor,
#     ):
#         ########################## Initialize ################################
#         assert concepts is None, "Concept input not yet implemented"
#         assert concept_ids is None, "Concept ID input not yet implemented"
#         concept_attention_dict = {}
#         ######################################################################
#         num_txt_tokens = ctx.shape[1]

#         timestep_emb = timestep_embedding(timesteps, 256)
#         vec = self.time_in(timestep_emb)
#         guidance_emb = timestep_embedding(guidance, 256)
#         vec = vec + self.guidance_in(guidance_emb)

#         double_block_mod_img = self.double_stream_modulation_img(vec)
#         double_block_mod_txt = self.double_stream_modulation_txt(vec)
#         single_block_mod, _ = self.single_stream_modulation(vec)

#         img = self.img_in(x)
#         txt = self.txt_in(ctx)
#         # ######################### Apply concept input ############################
#         # concepts = self.txt_in(concepts)
#         # ##########################################################################

#         pe_x = self.pe_embedder(x_ids)
#         pe_ctx = self.pe_embedder(ctx_ids)
#         # ################### Apply concept positional encodings ###################
#         # pe_concept = self.pe_embedder(concept_ids)
#         # ##########################################################################

#         for block in self.double_blocks:
#             # ############## Modified Double Stream Block Call ##################
#             # img, txt, _, block_concept_attention_dict = block(
#             #     img,
#             #     txt,
#             #     None,  # concepts,  # Adding concept tensor input
#             #     pe_x,
#             #     pe_ctx,
#             #     None,  # pe_concept,  # Adding concept positional encodings
#             #     double_block_mod_img,
#             #     double_block_mod_txt,
#             # )
#             # ############## End Modified Double Stream Block Call ##############
#             ############## Modified Double Stream Block Call ##################
#             img, txt = block(
#                 img,
#                 txt,
#                 pe_x,
#                 pe_ctx,
#                 double_block_mod_img,
#                 double_block_mod_txt,
#             )
#             ############## End Modified Double Stream Block Call ##############

#         ############## Perform concept attention projections here ##############
#         # Perform projections
#         # Add concept attention maps to concept_attention_dict
#         ########################################################################

#         img = torch.cat((txt, img), dim=1)
#         pe = torch.cat((pe_ctx, pe_x), dim=2)

#         for i, block in enumerate(self.single_blocks):
#             img = block(
#                 img,
#                 pe,
#                 single_block_mod,
#             )

#         img = img[:, num_txt_tokens:, ...]

#         img = self.final_layer(img, vec)
#         return img, concept_attention_dict


class ModifiedFlux2(nn.Module):
    def __init__(self, params: Flux2Params):
        super().__init__()

        self.in_channels = params.in_channels
        self.out_channels = params.in_channels
        if params.hidden_size % params.num_heads != 0:
            raise ValueError(
                f"Hidden size {params.hidden_size} must be divisible by num_heads {params.num_heads}"
            )
        pe_dim = params.hidden_size // params.num_heads
        if sum(params.axes_dim) != pe_dim:
            raise ValueError(
                f"Got {params.axes_dim} but expected positional dim {pe_dim}"
            )
        self.hidden_size = params.hidden_size
        self.num_heads = params.num_heads
        self.pe_embedder = EmbedND(
            dim=pe_dim, theta=params.theta, axes_dim=params.axes_dim
        )
        self.img_in = nn.Linear(self.in_channels, self.hidden_size, bias=False)
        self.time_in = MLPEmbedder(
            in_dim=256, hidden_dim=self.hidden_size, disable_bias=True
        )
        self.guidance_in = MLPEmbedder(
            in_dim=256, hidden_dim=self.hidden_size, disable_bias=True
        )
        self.txt_in = nn.Linear(params.context_in_dim, self.hidden_size, bias=False)

        self.double_blocks = nn.ModuleList(
            [
                ModifiedDoubleStreamBlock(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=params.mlp_ratio,
                )
                for _ in range(params.depth)
            ]
        )

        self.single_blocks = nn.ModuleList(
            [
                SingleStreamBlock(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=params.mlp_ratio,
                )
                for _ in range(params.depth_single_blocks)
            ]
        )

        self.double_stream_modulation_img = Modulation(
            self.hidden_size,
            double=True,
            disable_bias=True,
        )
        self.double_stream_modulation_txt = Modulation(
            self.hidden_size,
            double=True,
            disable_bias=True,
        )
        self.single_stream_modulation = Modulation(
            self.hidden_size, double=False, disable_bias=True
        )

        self.final_layer = LastLayer(
            self.hidden_size,
            self.out_channels,
        )

    def forward(
        self,
        x: Tensor,
        x_ids: Tensor,
        timesteps: Tensor,
        ctx: Tensor,
        ctx_ids: Tensor,
        guidance: Tensor,
        concepts: Tensor | None,
        concept_ids: Tensor | None,
    ):
        num_txt_tokens = ctx.shape[1]

        timestep_emb = timestep_embedding(timesteps, 256)
        vec = self.time_in(timestep_emb)
        guidance_emb = timestep_embedding(guidance, 256)
        vec = vec + self.guidance_in(guidance_emb)

        double_block_mod_img = self.double_stream_modulation_img(vec)
        double_block_mod_txt = self.double_stream_modulation_txt(vec)
        single_block_mod, _ = self.single_stream_modulation(vec)

        img = self.img_in(x)
        txt = self.txt_in(ctx)
        ########################### Apply concept input ############################
        if concepts is not None:
            concepts = self.txt_in(concepts)
        ##########################################################################

        pe_x = self.pe_embedder(x_ids)
        pe_ctx = self.pe_embedder(ctx_ids)
        ################### Apply concept positional encodings ###################
        pe_concept = self.pe_embedder(concept_ids)
        ##########################################################################
        concept_attention_dicts = []

        for block in self.double_blocks:
            img, txt, concepts, current_concept_attention_dict = block(
                img,
                txt,
                concepts,
                pe_x,
                pe_ctx,
                pe_concept,
                double_block_mod_img,
                double_block_mod_txt,
            )
            # Save the concept attention dict from this block
            concept_attention_dicts.append(current_concept_attention_dict)

        img = torch.cat((txt, img), dim=1)
        pe = torch.cat((pe_ctx, pe_x), dim=2)

        for i, block in enumerate(self.single_blocks):
            img = block(
                img,
                pe,
                single_block_mod,
            )

        img = img[:, num_txt_tokens:, ...]

        img = self.final_layer(img, vec)
        return img, concept_attention_dicts
