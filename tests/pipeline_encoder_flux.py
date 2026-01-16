from typing import Union
import PIL
import torch
import torch.nn as nn

# Pipeline-based encoder imports
from concept_attention import ConceptAttentionFluxPipeline
from concept_attention.utils import embed_concepts


class PipelineConceptEncoder(nn.Module):
    """
    Concept encoder built on top of ConceptAttentionFluxPipeline.

    This encoder wraps the pipeline's encode_image() function and provides
    a simpler interface focused on extracting concept and image vectors.
    """

    def __init__(
        self,
        model_name: str = "flux-schnell",
        device: str = "cuda:0",
        offload: bool = False,
    ):
        super().__init__()
        self.model_name = model_name
        self.device = device
        self.offload = offload

        # Initialize the pipeline
        self.pipeline = ConceptAttentionFluxPipeline(
            model_name=model_name,
            offload_model=offload,
            device=device,
        )

    @property
    def flux_generator(self):
        """Access to underlying flux generator for advanced usage."""
        return self.pipeline.flux_generator

    @torch.no_grad()
    def encode(
        self,
        image: Union[PIL.Image.Image, list[PIL.Image.Image]],
        concepts: list[str],
        prompt: str = "",
        width: int = 1024,
        height: int = 1024,
        layer_indices: list[int] = [15, 16, 17, 18],
        num_samples: int = 1,
        num_steps: int = 4,
        noise_timestep: int = 2,
        seed: int = 0,
        stop_after_multimodal_attentions: bool = True,
    ):
        """
        Encode an image and extract concept/image vectors.

        Args:
            image: PIL image or list of PIL images
            concepts: List of concept strings
            prompt: Optional text prompt
            width: Image width
            height: Image height
            layer_indices: Which transformer layers to extract vectors from
            num_samples: Number of noise samples to average
            num_steps: Number of diffusion steps
            noise_timestep: Which timestep to add noise at
            seed: Random seed
            stop_after_multimodal_attentions: Stop early for efficiency

        Returns:
            concept_vectors: Tensor of shape (layers, batch, concepts, dim)
            image_vectors: Tensor of shape (layers, batch, patches, dim)
        """
        # Use pipeline's encode_image with cache_vectors=True
        output = self.pipeline.encode_image(
            image=image,
            concepts=concepts,
            prompt=prompt,
            width=width,
            height=height,
            layer_indices=layer_indices,
            num_samples=num_samples,
            num_steps=num_steps,
            noise_timestep=noise_timestep,
            device=self.device,
            return_pil_heatmaps=False,  # We just want vectors
            seed=seed,
            stop_after_multi_modal_attentions=stop_after_multimodal_attentions,
            cache_vectors=True,
        )

        return output.concept_output_vectors, output.image_output_vectors

    @torch.no_grad()
    def encode_concepts_to_input_space(
        self,
        concepts: list[str],
    ):
        """
        Pre-encode concepts to input space embeddings.

        Args:
            concepts: List of concept strings

        Returns:
            concept_embeddings: Tensor of shape (1, num_concepts, embed_dim)
            concept_ids: Tensor of shape (1, num_concepts, 3)
            concept_vec: CLIP vector
        """
        return embed_concepts(
            self.flux_generator.clip,
            self.flux_generator.t5,
            concepts,
        )

    def forward(
        self,
        image: Union[PIL.Image.Image, list[PIL.Image.Image]],
        concepts: list[str],
        **kwargs,
    ):
        """Forward pass - alias for encode()."""
        return self.encode(image, concepts, **kwargs)
