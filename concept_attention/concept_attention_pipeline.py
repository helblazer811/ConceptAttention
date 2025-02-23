"""
    Wrapper pipeline for concept attention. 
"""
from dataclasses import dataclass
import PIL
import numpy as np
import matplotlib.pyplot as plt
import torch
import einops

from concept_attention.binary_segmentation_baselines.raw_cross_attention import RawCrossAttentionBaseline, RawCrossAttentionSegmentationModel
from concept_attention.binary_segmentation_baselines.raw_output_space import RawOutputSpaceBaseline, RawOutputSpaceSegmentationModel
from concept_attention.image_generator import FluxGenerator

@dataclass
class ConceptAttentionPipelineOutput():
    image: PIL.Image.Image | np.ndarray
    concept_heatmaps: list[PIL.Image.Image]
    cross_attention_maps: list[PIL.Image.Image]

class ConceptAttentionFluxPipeline():
    """
        This is an object that allows you to generate images with flux, and
        'encode' images with flux.  
    """

    def __init__(
        self, 
        model_name: str = "flux-schnell", 
        offload_model=False,
        device="cuda:0"
    ):
        self.model_name = model_name
        self.offload_model = offload_model
        # Load the generator
        self.flux_generator = FluxGenerator(
            model_name=model_name,
            offload=offload_model,
            device=device
        )

    @torch.no_grad()
    def generate_image(
        self, 
        prompt: str,
        concepts: list[str],
        width: int = 1024,
        height: int = 1024,
        return_cross_attention = False,
        layer_indices = list(range(15, 19)),
        return_pil_heatmaps = True,
        seed: int = 0,
        num_inference_steps: int = 4,
        guidance: float = 0.0,
        timesteps=None,
        softmax: bool = True,
        cmap="plasma"
    ) -> ConceptAttentionPipelineOutput:
        """
            Generate an image with flux, given a list of concepts.
        """
        assert return_cross_attention is False, "Not supported yet"
        assert all([layer_index >= 0 and layer_index < 19 for layer_index in layer_indices]), "Invalid layer index"
        assert height == width, "Height and width must be the same for now"

        if timesteps is None:
            timesteps = list(range(num_inference_steps))
        # Run the raw output space object
        image, cross_attention_maps, concept_heatmaps = self.flux_generator.generate_image(
            width=width,
            height=height,
            prompt=prompt,
            num_steps=num_inference_steps,
            concepts=concepts,
            seed=seed,
            guidance=guidance,
        )
        # Concept heamaps extraction
        if softmax:
            concept_heatmaps = torch.nn.functional.softmax(concept_heatmaps, dim=-2)

        concept_heatmaps = concept_heatmaps[:, layer_indices]
        concept_heatmaps = einops.reduce(
            concept_heatmaps,
            "time layers batch concepts patches -> batch concepts patches",
            reduction="mean"
        )
        concept_heatmaps = einops.rearrange(
            concept_heatmaps,
            "batch concepts (h w) -> batch concepts h w",
            h=64,
            w=64
        )
        # Cross attention maps 
        if softmax:
            cross_attention_maps = torch.nn.functional.softmax(cross_attention_maps, dim=-2)

        cross_attention_maps = cross_attention_maps[:, layer_indices]
        cross_attention_maps = einops.reduce(
            cross_attention_maps,
            "time layers batch concepts patches -> batch concepts patches",
            reduction="mean"
        )
        cross_attention_maps = einops.rearrange(
            cross_attention_maps,
            "batch concepts (h w) -> batch concepts h w",
            h=64,
            w=64
        )
        
        concept_heatmaps = concept_heatmaps.to(torch.float32).detach().cpu().numpy()[0]
        cross_attention_maps = cross_attention_maps.to(torch.float32).detach().cpu().numpy()[0]
        # Convert the torch heatmaps to PIL images.
        if return_pil_heatmaps:
            # Convert to a matplotlib color scheme
            colored_heatmaps = []
            for concept_heatmap in concept_heatmaps:
                concept_heatmap = (concept_heatmap - concept_heatmap.min()) / (concept_heatmap.max() - concept_heatmap.min())
                colored_heatmap = plt.get_cmap(cmap)(concept_heatmap)
                rgb_image = (colored_heatmap[:, :, :3] * 255).astype(np.uint8)
                colored_heatmaps.append(rgb_image)

            concept_heatmaps = [PIL.Image.fromarray(concept_heatmap) for concept_heatmap in colored_heatmaps]

            colored_cross_attention_maps = []
            for cross_attention_map in cross_attention_maps:
                cross_attention_map = (cross_attention_map - cross_attention_map.min()) / (cross_attention_map.max() - cross_attention_map.min())
                colored_cross_attention_map = plt.get_cmap(cmap)(cross_attention_map)
                rgb_image = (colored_cross_attention_map[:, :, :3] * 255).astype(np.uint8)
                colored_cross_attention_maps.append(rgb_image)

            cross_attention_maps = [PIL.Image.fromarray(cross_attention_map) for cross_attention_map in colored_cross_attention_maps]

        return ConceptAttentionPipelineOutput(
            image=image,
            concept_heatmaps=concept_heatmaps,
            cross_attention_maps=cross_attention_maps
        )

    # def encode_image(
    #     self,
    #     image: PIL.Image.Image,
    #     concepts: list[str],
    #     prompt: str = "", # Optional
    #     width: int = 1024,
    #     height: int = 1024,
    #     return_cross_attention = False,
    #     layer_indices = list(range(15, 19)),
    #     num_samples: int = 1,
    #     device: str = "cuda:0",
    #     return_pil_heatmaps: bool = True,
    #     seed: int = 0,
    #     cmap="plasma"
    # ) -> ConceptAttentionPipelineOutput:
    #     """
    #         Encode an image with flux, given a list of concepts.
    #     """
    #     assert return_cross_attention is False, "Not supported yet"
    #     assert all([layer_index >= 0 and layer_index < 19 for layer_index in layer_indices]), "Invalid layer index"
    #     assert height == width, "Height and width must be the same for now"
    #     # Run the raw output space object
    #     concept_heatmaps, _ = self.output_space_segmentation_model.segment_individual_image(
    #         image=image,
    #         concepts=concepts,
    #         caption=prompt,
    #         device=device,
    #         softmax=True,
    #         layers=layer_indices,
    #         num_samples=num_samples,
    #         height=height,
    #         width=width
    #     )
    #     concept_heatmaps = concept_heatmaps.detach().cpu().numpy().squeeze()
       
    #     # Convert the torch heatmaps to PIL images. 
    #     if return_pil_heatmaps:
    #         min_val = concept_heatmaps.min()
    #         max_val = concept_heatmaps.max()
    #         # Convert to a matplotlib color scheme
    #         colored_heatmaps = []
    #         for concept_heatmap in concept_heatmaps:
    #             # concept_heatmap = (concept_heatmap - concept_heatmap.min()) / (concept_heatmap.max() - concept_heatmap.min())
    #             concept_heatmap = (concept_heatmap - min_val) / (max_val - min_val)
    #             colored_heatmap = plt.get_cmap(cmap)(concept_heatmap)
    #             rgb_image = (colored_heatmap[:, :, :3] * 255).astype(np.uint8)
    #             colored_heatmaps.append(rgb_image)

    #         concept_heatmaps = [PIL.Image.fromarray(concept_heatmap) for concept_heatmap in colored_heatmaps]

    #     return ConceptAttentionPipelineOutput(
    #         image=image,
    #         concept_heatmaps=concept_heatmaps
    #     )

