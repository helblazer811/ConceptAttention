"""
    Wrapper pipeline for Flux 2 concept attention.
"""
from dataclasses import dataclass
import os
import sys

import PIL
import PIL.Image
import numpy as np
import matplotlib.pyplot as plt
import torch
import einops
from torch import Tensor
from einops import rearrange
from PIL import Image
from safetensors.torch import load_file as load_sft
from tqdm import tqdm
import huggingface_hub

from concept_attention.flux2.flux2.src.flux2.model import Flux2Params
from concept_attention.flux2.flux2.src.flux2.sampling import (
    batched_prc_img,
    batched_prc_txt,
    get_schedule,
    scatter_ids,
)
from concept_attention.flux2.flux2.src.flux2.util import load_ae, load_mistral_small_embedder
from concept_attention.flux2.modified_dit import ModifiedFlux2


@dataclass
class ConceptAttentionPipelineOutput:
    """Output from the ConceptAttentionFlux2Pipeline."""
    image: PIL.Image.Image | np.ndarray
    concept_heatmaps: list[PIL.Image.Image]
    cross_attention_maps: list[PIL.Image.Image]


def compute_heatmaps_from_attention_dicts(
    concept_attention_dicts: list,
    num_concepts: int,
    width: int,
    height: int,
    layer_indices: list[int],
    timestep_indices: list[int],
    key: str = "concept_scores",
    softmax: bool = True,
    softmax_temperature: float = 1.0,
) -> torch.Tensor:
    """
    Compute spatial heatmaps from concept attention dictionaries.

    Args:
        concept_attention_dicts: List of attention dicts per timestep
        num_concepts: Number of concepts
        width: Image width in pixels
        height: Image height in pixels
        layer_indices: Which layers/blocks to average over
        timestep_indices: Which timesteps to average over
        key: Which attention key to use ("concept_scores" or "cross_attention_scores")
        softmax: Whether to apply softmax normalization
        softmax_temperature: Temperature for softmax

    Returns:
        Tensor of shape (num_concepts, h, w) with spatial heatmaps
    """
    # Stack concept attention over time and blocks
    selected_concept_attention = []
    for t_idx in timestep_indices:
        if t_idx >= len(concept_attention_dicts):
            continue
        time_step_dicts = concept_attention_dicts[t_idx]
        selected_blocks = []
        for b_idx in layer_indices:
            if b_idx >= len(time_step_dicts):
                continue
            selected_blocks.append(time_step_dicts[b_idx][key])
        if selected_blocks:
            selected_blocks = torch.stack(selected_blocks)
            selected_concept_attention.append(selected_blocks)

    if not selected_concept_attention:
        raise ValueError("No valid attention data found for the specified indices")

    selected_concept_attention = torch.stack(selected_concept_attention)

    # Average over time and blocks
    avg_concept_scores = einops.reduce(
        selected_concept_attention,
        "time blocks batch num_concepts num_image_tokens -> batch num_concepts num_image_tokens",
        "mean",
    )
    avg_concept_scores = avg_concept_scores[0]  # Remove batch dim

    # Reshape to spatial grid (16 pixels per latent token)
    num_image_tokens_h = height // 16
    num_image_tokens_w = width // 16
    avg_concept_scores = einops.rearrange(
        avg_concept_scores,
        "num_concepts (h w) -> num_concepts h w",
        h=num_image_tokens_h,
        w=num_image_tokens_w,
    )

    # Apply softmax normalization across concepts
    if softmax:
        avg_concept_scores = torch.softmax(avg_concept_scores / softmax_temperature, dim=0)

    return avg_concept_scores


def heatmaps_to_pil_images(
    heatmaps: torch.Tensor,
    width: int,
    height: int,
    cmap: str = "plasma",
) -> list[PIL.Image.Image]:
    """
    Convert tensor heatmaps to colored PIL images.

    Args:
        heatmaps: Tensor of shape (num_concepts, h, w)
        width: Target width
        height: Target height
        cmap: Matplotlib colormap name

    Returns:
        List of PIL images
    """
    heatmaps_np = heatmaps.cpu().float().numpy()
    global_min = heatmaps_np.min()
    global_max = heatmaps_np.max()

    colormap = plt.get_cmap(cmap)
    pil_images = []

    for concept_heatmap in heatmaps_np:
        # Normalize to [0, 1]
        normalized = (concept_heatmap - global_min) / (global_max - global_min + 1e-8)
        # Apply colormap
        colored = colormap(normalized)
        rgb_image = (colored[:, :, :3] * 255).astype(np.uint8)
        pil_img = PIL.Image.fromarray(rgb_image)
        # Resize to target dimensions
        pil_img = pil_img.resize((width, height), resample=PIL.Image.NEAREST)
        pil_images.append(pil_img)

    return pil_images


class ConceptAttentionFlux2Pipeline:
    """
    Pipeline for generating images with Flux 2 and extracting concept attention heatmaps.

    This mirrors the interface of ConceptAttentionFluxPipeline for Flux 1.
    """

    def __init__(
        self,
        model_name: str = "flux.2-dev",
        device: str = "cuda:0",
        offload_model: bool = False,
    ):
        """
        Initialize the Flux 2 pipeline.

        Args:
            model_name: Model name (currently only "flux.2-dev" supported)
            device: Device to run on
            offload_model: Whether to offload models to CPU when not in use
        """
        self.model_name = model_name
        self.device = device
        self.offload_model = offload_model

        print("Loading Flux 2 models...")

        # Load Mistral text embedder
        self.mistral = load_mistral_small_embedder()
        self.mistral.eval()

        # Load flow model
        self._load_flow_model()

        # Load VAE autoencoder
        self.ae = load_ae(model_name)
        self.ae.eval()

        print("Flux 2 models loaded successfully!")

    def _load_flow_model(self):
        """Load the ModifiedFlux2 flow model."""
        repo_id = "black-forest-labs/FLUX.2-dev"
        filename = "flux2-dev.safetensors"

        if "FLUX2_MODEL_PATH" in os.environ:
            weight_path = os.environ["FLUX2_MODEL_PATH"]
        else:
            try:
                weight_path = huggingface_hub.hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    repo_type="model",
                )
            except huggingface_hub.errors.RepositoryNotFoundError:
                print(f"Failed to access {repo_id}. Check your access permissions.")
                sys.exit(1)

        with torch.device("meta"):
            self.model = ModifiedFlux2(Flux2Params()).to(torch.bfloat16)

        print(f"Loading weights from {weight_path}")
        sd = load_sft(weight_path, device=str(self.device))
        self.model.load_state_dict(sd, strict=False, assign=True)
        self.model = self.model.to(self.device)

    def _denoise(
        self,
        img: Tensor,
        img_ids: Tensor,
        txt: Tensor,
        txt_ids: Tensor,
        concepts: Tensor,
        concept_ids: Tensor,
        timesteps: list[float],
        guidance: float,
    ) -> tuple[Tensor, list]:
        """
        Denoising loop that tracks concept attention at each step.

        Returns:
            Tuple of (denoised_img, concept_attention_dicts)
        """
        guidance_vec = torch.full(
            (img.shape[0],), guidance, device=img.device, dtype=img.dtype
        )
        concept_attention_dicts = []

        for t_curr, t_prev in tqdm(
            zip(timesteps[:-1], timesteps[1:]),
            total=len(timesteps) - 1,
            desc="Denoising"
        ):
            t_vec = torch.full(
                (img.shape[0],), t_curr, dtype=img.dtype, device=img.device
            )

            pred, current_concept_attention_dict = self.model(
                x=img,
                x_ids=img_ids,
                timesteps=t_vec,
                ctx=txt,
                ctx_ids=txt_ids,
                guidance=guidance_vec,
                concepts=concepts,
                concept_ids=concept_ids,
            )
            concept_attention_dicts.append(current_concept_attention_dict)
            img = img + (t_prev - t_curr) * pred

        return img, concept_attention_dicts

    @torch.no_grad()
    def generate_image(
        self,
        prompt: str,
        concepts: list[str],
        width: int = 2048,
        height: int = 2048,
        num_inference_steps: int = 28,
        guidance: float = 4.0,
        seed: int = 0,
        layer_indices: list[int] = None,
        timesteps: list[int] = None,
        return_pil_heatmaps: bool = True,
        softmax: bool = True,
        softmax_temperature: float = 1000.0,
        cmap: str = "plasma",
    ) -> ConceptAttentionPipelineOutput:
        """
        Generate an image with Flux 2 and extract concept attention heatmaps.

        Args:
            prompt: Text prompt for generation
            concepts: List of concept words to track attention for
            width: Output image width (should be divisible by 16)
            height: Output image height (should be divisible by 16)
            num_inference_steps: Number of denoising steps
            guidance: Guidance scale
            seed: Random seed
            layer_indices: Which transformer blocks to average over (default: last 3)
            timesteps: Which timesteps to average over (default: last 30%)
            return_pil_heatmaps: Whether to return PIL images (True) or numpy arrays
            softmax: Whether to apply softmax normalization
            softmax_temperature: Temperature for softmax
            cmap: Matplotlib colormap for heatmaps

        Returns:
            ConceptAttentionPipelineOutput with image, concept_heatmaps, and cross_attention_maps
        """
        # Default layer indices (last 3 of 8 double blocks)
        if layer_indices is None:
            layer_indices = [5, 6, 7]

        # Default timestep indices (last 30% of steps)
        if timesteps is None:
            start_idx = int(num_inference_steps * 0.7)
            timesteps = list(range(start_idx, num_inference_steps - 1))

        # Validate inputs
        assert all(0 <= idx < 8 for idx in layer_indices), "layer_indices must be in [0, 7]"

        # Encode text prompt
        ctx = self.mistral([prompt]).to(torch.bfloat16)
        ctx, ctx_ids = batched_prc_txt(ctx)

        # Encode concepts
        concept_embeddings = self.mistral(concepts).to(torch.bfloat16)
        concepts_tensor, concept_ids = batched_prc_txt(concept_embeddings)
        # Extract single token representation per concept (at position 510)
        concepts_tensor = concepts_tensor[:, 510].unsqueeze(0)
        concept_ids = concept_ids[:, 510].unsqueeze(0)

        # Offload text encoder if needed
        if self.offload_model:
            self.mistral = self.mistral.cpu()
            torch.cuda.empty_cache()
            self.model = self.model.to(self.device)

        # Create noise
        shape = (1, 128, height // 16, width // 16)
        generator = torch.Generator(device=self.device).manual_seed(seed)
        randn = torch.randn(
            shape, generator=generator, dtype=torch.bfloat16, device=self.device
        )
        x, x_ids = batched_prc_img(randn)

        # Denoise with concept tracking
        schedule = get_schedule(num_inference_steps, x.shape[1])
        x, concept_attention_dicts = self._denoise(
            x, x_ids, ctx, ctx_ids,
            concepts=concepts_tensor,
            concept_ids=concept_ids,
            timesteps=schedule,
            guidance=guidance,
        )

        # Offload flow model, load VAE
        if self.offload_model:
            self.model = self.model.cpu()
            torch.cuda.empty_cache()

        # Decode latents to image
        x = torch.cat(scatter_ids(x, x_ids)).squeeze(2)
        x = self.ae.decode(x).float()

        # Convert to PIL image
        x = x.clamp(-1, 1)
        x = rearrange(x[0], "c h w -> h w c")
        image = PIL.Image.fromarray((127.5 * (x + 1.0)).cpu().byte().numpy())

        # Compute concept heatmaps (output space attention)
        concept_heatmaps_tensor = compute_heatmaps_from_attention_dicts(
            concept_attention_dicts,
            num_concepts=len(concepts),
            width=width,
            height=height,
            layer_indices=layer_indices,
            timestep_indices=timesteps,
            key="concept_scores",
            softmax=softmax,
            softmax_temperature=softmax_temperature,
        )

        # Compute cross-attention heatmaps
        cross_attention_tensor = compute_heatmaps_from_attention_dicts(
            concept_attention_dicts,
            num_concepts=len(concepts),
            width=width,
            height=height,
            layer_indices=layer_indices,
            timestep_indices=timesteps,
            key="cross_attention_scores",
            softmax=softmax,
            softmax_temperature=softmax_temperature,
        )

        # Convert to PIL images if requested
        if return_pil_heatmaps:
            concept_heatmaps = heatmaps_to_pil_images(
                concept_heatmaps_tensor, width, height, cmap
            )
            cross_attention_maps = heatmaps_to_pil_images(
                cross_attention_tensor, width, height, cmap
            )
        else:
            concept_heatmaps = concept_heatmaps_tensor.cpu().numpy()
            cross_attention_maps = cross_attention_tensor.cpu().numpy()

        # Restore models if offloaded
        if self.offload_model:
            self.mistral = self.mistral.to(self.device)

        return ConceptAttentionPipelineOutput(
            image=image,
            concept_heatmaps=concept_heatmaps,
            cross_attention_maps=cross_attention_maps,
        )

    @torch.no_grad()
    def encode_image(
        self,
        image: PIL.Image.Image,
        concepts: list[str],
        prompt: str = "",
        width: int = 2048,
        height: int = 2048,
        layer_indices: list[int] = None,
        num_steps: int = 4,
        noise_timestep: int = 2,
        seed: int = 0,
        return_pil_heatmaps: bool = True,
        softmax: bool = True,
        softmax_temperature: float = 1000.0,
        cmap: str = "plasma",
    ) -> ConceptAttentionPipelineOutput:
        """
        Encode an existing image and extract concept attention heatmaps.

        Args:
            image: Input PIL image
            concepts: List of concept words to track attention for
            prompt: Optional text prompt describing the image
            width: Processing width (image will be resized)
            height: Processing height (image will be resized)
            layer_indices: Which transformer blocks to average over
            num_steps: Number of noise levels to use
            noise_timestep: Which noise level to add
            seed: Random seed for noise
            return_pil_heatmaps: Whether to return PIL images
            softmax: Whether to apply softmax normalization
            softmax_temperature: Temperature for softmax
            cmap: Matplotlib colormap for heatmaps

        Returns:
            ConceptAttentionPipelineOutput with original image, concept_heatmaps, and cross_attention_maps
        """
        # Default layer indices
        if layer_indices is None:
            layer_indices = [5, 6, 7]

        # Resize image to target dimensions
        image_resized = image.resize((width, height), PIL.Image.LANCZOS)

        # Encode text prompt and concepts
        ctx = self.mistral([prompt] if prompt else [""]).to(torch.bfloat16)
        ctx, ctx_ids = batched_prc_txt(ctx)

        concept_embeddings = self.mistral(concepts).to(torch.bfloat16)
        concepts_tensor, concept_ids = batched_prc_txt(concept_embeddings)
        concepts_tensor = concepts_tensor[:, 510].unsqueeze(0)
        concept_ids = concept_ids[:, 510].unsqueeze(0)

        if self.offload_model:
            self.mistral = self.mistral.cpu()
            torch.cuda.empty_cache()
            self.model = self.model.to(self.device)

        # Encode image to latent space
        img_array = np.array(image_resized).astype(np.float32) / 127.5 - 1.0
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
        img_tensor = img_tensor.to(self.device, dtype=torch.bfloat16)

        # Encode with VAE
        encoded_image = self.ae.encode(img_tensor)

        # Add noise
        schedule = get_schedule(num_steps, encoded_image.shape[2] * encoded_image.shape[3])
        t = schedule[noise_timestep]

        generator = torch.Generator(device=self.device).manual_seed(seed)
        noise = torch.randn_like(encoded_image, generator=generator)
        noisy_latent = (1 - t) * encoded_image + t * noise

        # Prepare for model
        x, x_ids = batched_prc_img(noisy_latent)

        # Run single forward pass
        t_vec = torch.full((1,), t, dtype=x.dtype, device=x.device)
        guidance_vec = torch.full((1,), 0.0, dtype=x.dtype, device=x.device)

        _, concept_attention_dict = self.model(
            x=x,
            x_ids=x_ids,
            timesteps=t_vec,
            ctx=ctx,
            ctx_ids=ctx_ids,
            guidance=guidance_vec,
            concepts=concepts_tensor,
            concept_ids=concept_ids,
        )

        # Wrap in list for compatibility with compute function
        concept_attention_dicts = [concept_attention_dict]

        # Compute heatmaps
        concept_heatmaps_tensor = compute_heatmaps_from_attention_dicts(
            concept_attention_dicts,
            num_concepts=len(concepts),
            width=width,
            height=height,
            layer_indices=layer_indices,
            timestep_indices=[0],
            key="concept_scores",
            softmax=softmax,
            softmax_temperature=softmax_temperature,
        )

        cross_attention_tensor = compute_heatmaps_from_attention_dicts(
            concept_attention_dicts,
            num_concepts=len(concepts),
            width=width,
            height=height,
            layer_indices=layer_indices,
            timestep_indices=[0],
            key="cross_attention_scores",
            softmax=softmax,
            softmax_temperature=softmax_temperature,
        )

        if return_pil_heatmaps:
            concept_heatmaps = heatmaps_to_pil_images(
                concept_heatmaps_tensor, width, height, cmap
            )
            cross_attention_maps = heatmaps_to_pil_images(
                cross_attention_tensor, width, height, cmap
            )
        else:
            concept_heatmaps = concept_heatmaps_tensor.cpu().numpy()
            cross_attention_maps = cross_attention_tensor.cpu().numpy()

        if self.offload_model:
            self.model = self.model.cpu()
            self.mistral = self.mistral.to(self.device)
            torch.cuda.empty_cache()

        return ConceptAttentionPipelineOutput(
            image=image_resized,
            concept_heatmaps=concept_heatmaps,
            cross_attention_maps=cross_attention_maps,
        )
