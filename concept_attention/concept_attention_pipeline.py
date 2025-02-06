"""
    Wrapper pipeline for concept attention. 
"""
from dataclasses import dataclass
import PIL
import numpy as np
import matplotlib.pyplot as plt

from concept_attention.binary_segmentation_baselines.raw_cross_attention import RawCrossAttentionBaseline, RawCrossAttentionSegmentationModel
from concept_attention.binary_segmentation_baselines.raw_output_space import RawOutputSpaceBaseline, RawOutputSpaceSegmentationModel
from concept_attention.image_generator import FluxGenerator

@dataclass
class ConceptAttentionPipelineOutput():
    image: PIL.Image.Image | np.ndarray
    concept_heatmaps: list[PIL.Image.Image]

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
        self.offload_model = False
        # Load the generator
        self.flux_generator = FluxGenerator(
            model_name=model_name,
            offload=offload_model,
            device=device
        )
        # Make a Raw Cross Attention Segmentation Model and Raw Output space segmentation model
        self.cross_attention_segmentation_model = RawCrossAttentionSegmentationModel(
            generator=self.flux_generator
        )
        self.output_space_segmentation_model = RawOutputSpaceSegmentationModel(
            generator=self.flux_generator
        )
        self.raw_output_space_generator = RawOutputSpaceBaseline(
            generator=self.flux_generator
        )
        self.raw_cross_attention_generator = RawCrossAttentionBaseline(
            generator=self.flux_generator
        )

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
        concept_heatmaps, image = self.raw_output_space_generator(
            prompt,
            concepts,
            seed=seed,
            num_steps=num_inference_steps,
            timesteps=timesteps,
            layers=layer_indices,
            softmax=softmax,
            height=width,
            width=width,
            guidance=guidance,
        )
        # Convert to numpy 
        concept_heatmaps = concept_heatmaps.detach().cpu().numpy()
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

        return ConceptAttentionPipelineOutput(
            image=image,
            concept_heatmaps=concept_heatmaps
        )

    def encode_image(
        self,
        image: PIL.Image.Image,
        concepts: list[str],
        prompt: str = "", # Optional
        width: int = 1024,
        height: int = 1024,
        return_cross_attention = False,
        layer_indices = list(range(15, 19)),
        num_samples: int = 1,
        device: str = "cuda:0",
        return_pil_heatmaps: bool = True,
        seed: int = 0,
        cmap="plasma"
    ) -> ConceptAttentionPipelineOutput:
        """
            Encode an image with flux, given a list of concepts.
        """
        assert return_cross_attention is False, "Not supported yet"
        assert all([layer_index >= 0 and layer_index < 19 for layer_index in layer_indices]), "Invalid layer index"
        assert height == width, "Height and width must be the same for now"
        # Run the raw output space object
        concept_heatmaps, _ = self.output_space_segmentation_model.segment_individual_image(
            image=image,
            concepts=concepts,
            caption=prompt,
            device=device,
            softmax=True,
            layers=layer_indices,
            num_samples=num_samples,
            height=height,
            width=width
        )
        concept_heatmaps = concept_heatmaps.detach().cpu().numpy()
       
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

        return ConceptAttentionPipelineOutput(
            image=image,
            concept_heatmaps=concept_heatmaps
        )

