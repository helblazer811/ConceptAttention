"""
Wrapper pipeline for SD3 concept attention with unified generate_image/encode_image interface.
"""
from dataclasses import dataclass
from typing import List, Optional, Union

import PIL.Image
import numpy as np
import torch

from concept_attention.sd3.pipeline import (
    CustomStableDiffusion3Pipeline,
    calculate_shift,
    retrieve_timesteps,
)
from concept_attention.sd3.dit_block import CustomSD3Transformer2DModel
from concept_attention.heatmap_utils import (
    compute_heatmaps_from_vectors,
    heatmaps_to_pil_images,
)
from diffusers.utils.torch_utils import randn_tensor


def retrieve_latents(encoder_output, generator=None, sample_mode="sample"):
    """Extract latents from VAE encoder output."""
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")


@dataclass
class ConceptAttentionPipelineOutput:
    """Output from the ConceptAttentionSD3Pipeline."""
    image: PIL.Image.Image
    concept_heatmaps: list[PIL.Image.Image]
    # Raw output vectors (only populated when cache_vectors=True)
    concept_output_vectors: torch.Tensor | None = None
    image_output_vectors: torch.Tensor | None = None


class ConceptAttentionSD3Pipeline:
    """
    Pipeline for generating/encoding images with SD3 and extracting concept attention heatmaps.

    This provides a unified interface matching ConceptAttentionFluxPipeline and
    ConceptAttentionFlux2Pipeline.
    """

    def __init__(
        self,
        model_name: str = "stabilityai/stable-diffusion-3.5-large-turbo",
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        """
        Initialize the SD3 pipeline.

        Args:
            model_name: HuggingFace model ID or path
            device: Device to run on
            dtype: Torch dtype for the model
        """
        self.model_name = model_name
        self.device = device
        self.dtype = dtype

        print(f"Loading transformer from {model_name}...")
        self.transformer = CustomSD3Transformer2DModel.from_pretrained(
            model_name,
            subfolder="transformer",
            torch_dtype=dtype
        )

        print("Loading pipeline...")
        self.pipe = CustomStableDiffusion3Pipeline.from_pretrained(
            model_name,
            transformer=self.transformer,
            torch_dtype=dtype
        ).to(device)

        print("SD3 pipeline loaded successfully!")

    def _compute_heatmaps(
        self,
        concept_vectors: torch.Tensor,
        image_vectors: torch.Tensor,
        height: int,
        width: int,
        layer_indices: Optional[List[int]] = None,
        softmax: bool = True,
        average_over_timesteps: bool = False,
    ) -> torch.Tensor:
        """
        Compute heatmaps from concept and image vectors.

        Args:
            concept_vectors: Shape (layers, batch, concepts, dims) or (timesteps, layers, batch, concepts, dims)
            image_vectors: Shape (layers, batch, pixels, dims) or (timesteps, layers, batch, pixels, dims)
            height: Image height in pixels
            width: Image width in pixels
            layer_indices: Which layers to use (default: all)
            softmax: Whether to apply softmax normalization
            average_over_timesteps: Whether vectors have timestep dimension

        Returns:
            Tensor of shape (num_concepts, h, w)
        """
        # Handle timestep dimension if present
        if average_over_timesteps and len(concept_vectors.shape) == 5:
            # Average across timesteps first
            concept_vectors = torch.mean(concept_vectors, dim=0)
            image_vectors = torch.mean(image_vectors, dim=0)

        # Shape is now (layers, batch, tokens, dims)
        num_layers = concept_vectors.shape[0]

        # Default to all layers
        if layer_indices is None:
            layer_indices = list(range(num_layers))

        # Drop the padding token (first token)
        concept_vectors = concept_vectors[:, :, 1:]

        # Normalize concept vectors
        concept_vectors = concept_vectors / (concept_vectors.norm(dim=-1, keepdim=True) + 1e-8)

        # Compute heatmaps via einsum
        # concept_vectors: (layers, batch, concepts, dims)
        # image_vectors: (layers, batch, pixels, dims)
        heatmaps = torch.einsum(
            "lbcd,lbpd->lbcp",
            concept_vectors.float(),
            image_vectors.float()
        )
        # heatmaps: (layers, batch, concepts, pixels)

        # Apply softmax across concepts
        if softmax:
            heatmaps = torch.nn.functional.softmax(heatmaps, dim=-2)

        # Select specified layers
        heatmaps = heatmaps[layer_indices]

        # Average across layers
        heatmaps = heatmaps.mean(dim=0)  # (batch, concepts, pixels)

        # Remove batch dimension (assuming batch=1)
        heatmaps = heatmaps[0]  # (concepts, pixels)

        # Reshape to spatial dimensions
        # For SD3 with 1024x1024 image: 64x64 tokens (patch_size=2, vae_scale_factor=8)
        h_tokens = height // (self.pipe.vae_scale_factor * self.pipe.patch_size)
        w_tokens = width // (self.pipe.vae_scale_factor * self.pipe.patch_size)

        heatmaps = heatmaps.view(-1, h_tokens, w_tokens)

        return heatmaps

    @torch.no_grad()
    def generate_image(
        self,
        prompt: str,
        concepts: List[str],
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 4,
        guidance_scale: float = 0.0,
        seed: Optional[int] = None,
        layer_indices: Optional[List[int]] = None,
        timestep_indices: Optional[List[int]] = None,
        return_pil_heatmaps: bool = True,
        softmax: bool = True,
        cmap: str = "plasma",
        cache_vectors: bool = True,
    ) -> ConceptAttentionPipelineOutput:
        """
        Generate an image with SD3 and extract concept attention heatmaps.

        Args:
            prompt: Text prompt for generation
            concepts: List of concept words to track attention for
            height: Output image height
            width: Output image width
            num_inference_steps: Number of denoising steps
            guidance_scale: Guidance scale (0.0 for turbo models)
            seed: Random seed (None for random)
            layer_indices: Which transformer layers to average over (default: all)
            timestep_indices: Which timesteps to track (default: all)
            return_pil_heatmaps: Whether to return PIL images or tensor
            softmax: Whether to apply softmax normalization
            cmap: Matplotlib colormap for heatmaps
            cache_vectors: Whether to cache raw output vectors

        Returns:
            ConceptAttentionPipelineOutput with image, concept_heatmaps, and optionally raw vectors
        """
        # Set seed if provided
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        # Run the pipeline
        result, concept_attention_outputs = self.pipe(
            prompt=prompt,
            concepts=concepts,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            generator=generator,
            cache_vectors=cache_vectors,
            layer_indices=layer_indices,
            timestep_indices=timestep_indices,
        )

        image = result.images[0]

        # Get raw vectors
        concept_vectors = concept_attention_outputs.get("concept_output_vectors")
        image_vectors = concept_attention_outputs.get("image_output_vectors")

        # Compute heatmaps
        if concept_vectors is not None and image_vectors is not None:
            heatmaps_tensor = self._compute_heatmaps(
                concept_vectors=concept_vectors,
                image_vectors=image_vectors,
                height=height,
                width=width,
                layer_indices=layer_indices,
                softmax=softmax,
                average_over_timesteps=True,
            )

            if return_pil_heatmaps:
                concept_heatmaps = heatmaps_to_pil_images(
                    heatmaps_tensor, width, height, cmap
                )
            else:
                concept_heatmaps = heatmaps_tensor.cpu()
        else:
            concept_heatmaps = []

        return ConceptAttentionPipelineOutput(
            image=image,
            concept_heatmaps=concept_heatmaps,
            concept_output_vectors=concept_vectors,
            image_output_vectors=image_vectors,
        )

    @torch.no_grad()
    def encode_image(
        self,
        image: PIL.Image.Image,
        prompt: str,
        concepts: List[str],
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 4,
        timestep_index: int = -2,
        layer_indices: Optional[List[int]] = None,
        seed: Optional[int] = None,
        return_pil_heatmaps: bool = True,
        softmax: bool = True,
        cmap: str = "plasma",
        cache_vectors: bool = True,
    ) -> ConceptAttentionPipelineOutput:
        """
        Encode an existing image and extract concept attention heatmaps.

        Args:
            image: Input PIL image
            prompt: Text caption describing the image
            concepts: List of concept words to track attention for
            height: Processing height (image will be resized)
            width: Processing width (image will be resized)
            num_inference_steps: Number of noise levels for scheduler
            timestep_index: Which timestep to use for encoding (negative indexing supported)
            layer_indices: Which transformer layers to average over (default: all)
            seed: Random seed for noise
            return_pil_heatmaps: Whether to return PIL images or tensor
            softmax: Whether to apply softmax normalization
            cmap: Matplotlib colormap for heatmaps
            cache_vectors: Whether to cache raw output vectors

        Returns:
            ConceptAttentionPipelineOutput with resized image, concept_heatmaps, and raw vectors
        """
        device = self.device
        dtype = self.dtype

        # Preprocess the image
        image_tensor = self.pipe.image_processor.preprocess(
            image, height=height, width=width
        ).to(device=device, dtype=dtype)

        # Encode with VAE
        init_latents = retrieve_latents(self.pipe.vae.encode(image_tensor))

        # Set up scheduler
        scheduler_kwargs = {}
        if self.pipe.scheduler.config.get("use_dynamic_shifting", None):
            image_seq_len = (
                int(height) // self.pipe.vae_scale_factor // self.pipe.transformer.config.patch_size
            ) * (
                int(width) // self.pipe.vae_scale_factor // self.pipe.transformer.config.patch_size
            )
            mu = calculate_shift(
                image_seq_len,
                self.pipe.scheduler.config.get("base_image_seq_len", 256),
                self.pipe.scheduler.config.get("max_image_seq_len", 4096),
                self.pipe.scheduler.config.get("base_shift", 0.5),
                self.pipe.scheduler.config.get("max_shift", 1.16),
            )
            scheduler_kwargs["mu"] = mu

        timesteps, _ = retrieve_timesteps(
            self.pipe.scheduler, num_inference_steps, device, sigmas=None, **scheduler_kwargs
        )
        latent_timestep = self.pipe.scheduler.timesteps[timestep_index * self.pipe.scheduler.order]
        latent_timestep = latent_timestep.unsqueeze(0)

        # Scale latents
        init_latents = (init_latents - self.pipe.vae.config.shift_factor) * self.pipe.vae.config.scaling_factor
        init_latents = torch.cat([init_latents], dim=0)

        # Add noise
        generator = None
        if seed is not None:
            generator = torch.Generator(device=device).manual_seed(seed)
        noise = randn_tensor(init_latents.shape, generator=generator, device=device, dtype=dtype)
        noisy_latents = self.pipe.scheduler.scale_noise(init_latents, latent_timestep, noise)
        noisy_latents = noisy_latents.to(device=device, dtype=dtype)

        # Encode the prompt
        prompt_embeds, _, pooled_prompt_embeds, _ = self.pipe.encode_prompt(prompt, None, None)

        # Encode the concepts
        concept_embeds = self.pipe.encode_concepts(concepts)

        # Get timestep
        timestep = timesteps[timestep_index].expand(noisy_latents.shape[0])

        # Run the transformer
        noise_pred, concept_attention_outputs = self.pipe.transformer(
            hidden_states=noisy_latents,
            timestep=timestep,
            encoder_hidden_states=prompt_embeds,
            concept_hidden_states=concept_embeds,
            pooled_projections=pooled_prompt_embeds,
            joint_attention_kwargs=None,
            return_dict=False,
            cache_vectors=cache_vectors,
            layer_indices=layer_indices,
        )

        # Get raw vectors
        concept_vectors = concept_attention_outputs.get("concept_output_vectors")
        image_vectors = concept_attention_outputs.get("image_output_vectors")

        # Compute heatmaps
        if concept_vectors is not None and image_vectors is not None:
            heatmaps_tensor = self._compute_heatmaps(
                concept_vectors=concept_vectors,
                image_vectors=image_vectors,
                height=height,
                width=width,
                layer_indices=layer_indices,
                softmax=softmax,
                average_over_timesteps=False,
            )

            if return_pil_heatmaps:
                concept_heatmaps = heatmaps_to_pil_images(
                    heatmaps_tensor, width, height, cmap
                )
            else:
                concept_heatmaps = heatmaps_tensor.cpu()
        else:
            concept_heatmaps = []

        # Get the processed image back
        processed_image = self.pipe.image_processor.postprocess(
            image_tensor, output_type="pil"
        )[0]

        return ConceptAttentionPipelineOutput(
            image=processed_image,
            concept_heatmaps=concept_heatmaps,
            concept_output_vectors=concept_vectors,
            image_output_vectors=image_vectors,
        )
