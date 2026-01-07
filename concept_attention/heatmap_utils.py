"""
Shared heatmap computation and visualization utilities for all concept attention pipelines.
"""
from typing import List, Optional, Union
import numpy as np
import PIL.Image
import matplotlib.pyplot as plt
import torch
import einops


def aggregate_attention_over_timesteps(
    attention_dicts: List[dict],
    layer_indices: List[int],
    timestep_indices: List[int],
    key: str = "concept_scores",
) -> torch.Tensor:
    """
    Aggregate attention scores across timesteps and layers.

    Args:
        attention_dicts: List of attention dicts, one per timestep. Each dict
            contains a list of per-layer attention dicts.
        layer_indices: Which layers/blocks to average over.
        timestep_indices: Which timesteps to average over.
        key: Which attention key to use (e.g., "concept_scores", "cross_attention_scores").

    Returns:
        Tensor of shape (batch, num_concepts, num_image_tokens) with aggregated scores.
    """
    selected_attention = []
    for t_idx in timestep_indices:
        if t_idx >= len(attention_dicts):
            continue
        time_step_dicts = attention_dicts[t_idx]
        selected_blocks = []
        for b_idx in layer_indices:
            if b_idx >= len(time_step_dicts):
                continue
            selected_blocks.append(time_step_dicts[b_idx][key])
        if selected_blocks:
            selected_blocks = torch.stack(selected_blocks)
            selected_attention.append(selected_blocks)

    if not selected_attention:
        raise ValueError("No valid attention data found for the specified indices")

    selected_attention = torch.stack(selected_attention)

    # Average over time and blocks
    avg_scores = einops.reduce(
        selected_attention,
        "time blocks batch num_concepts num_image_tokens -> batch num_concepts num_image_tokens",
        "mean",
    )

    return avg_scores


def compute_heatmaps_from_attention_dicts(
    attention_dicts: List[dict],
    num_concepts: int,
    width: int,
    height: int,
    layer_indices: List[int],
    timestep_indices: List[int],
    key: str = "concept_scores",
    softmax: bool = True,
    softmax_temperature: float = 1.0,
    pixels_per_latent: int = 16,
) -> torch.Tensor:
    """
    Compute spatial heatmaps from concept attention dictionaries.

    This is the main function for converting raw attention scores into
    spatial heatmaps that can be visualized.

    Args:
        attention_dicts: List of attention dicts per timestep.
        num_concepts: Number of concepts.
        width: Image width in pixels.
        height: Image height in pixels.
        layer_indices: Which layers/blocks to average over.
        timestep_indices: Which timesteps to average over.
        key: Which attention key to use ("concept_scores" or "cross_attention_scores").
        softmax: Whether to apply softmax normalization across concepts.
        softmax_temperature: Temperature for softmax.
        pixels_per_latent: Number of image pixels per latent token (16 for Flux).

    Returns:
        Tensor of shape (num_concepts, h, w) with spatial heatmaps.
    """
    # Aggregate attention across timesteps and layers
    avg_concept_scores = aggregate_attention_over_timesteps(
        attention_dicts,
        layer_indices=layer_indices,
        timestep_indices=timestep_indices,
        key=key,
    )
    avg_concept_scores = avg_concept_scores[0]  # Remove batch dim

    # Reshape to spatial grid
    num_image_tokens_h = height // pixels_per_latent
    num_image_tokens_w = width // pixels_per_latent
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


def compute_heatmaps_from_vectors(
    image_vectors: torch.Tensor,
    concept_vectors: torch.Tensor,
    layer_indices: List[int],
    timestep_indices: List[int],
    height_tokens: int = 64,
    width_tokens: int = 64,
    softmax: bool = True,
    normalize_concepts: bool = False,
) -> torch.Tensor:
    """
    Compute heatmaps from image and concept vectors using einsum.

    This is an alternative method that works with raw vectors rather than
    pre-computed attention scores.

    Args:
        image_vectors: Image representations, shape varies by implementation.
        concept_vectors: Concept representations, shape varies by implementation.
        layer_indices: Which layers to average over.
        timestep_indices: Which timesteps to average over.
        height_tokens: Height in latent tokens.
        width_tokens: Width in latent tokens.
        softmax: Whether to apply softmax normalization.
        normalize_concepts: Whether to apply linear normalization to concepts.

    Returns:
        Tensor of shape (batch, num_concepts, height_tokens, width_tokens).
    """
    # Handle head dimension if present
    if len(image_vectors.shape) == 6:
        image_vectors = einops.rearrange(
            image_vectors,
            "time layers batch head patches dim -> time layers batch patches (head dim)"
        )
        concept_vectors = einops.rearrange(
            concept_vectors,
            "time layers batch head concepts dim -> time layers batch concepts (head dim)"
        )

    # Apply linear normalization to concepts if requested
    if normalize_concepts:
        # Simple linear normalization along the concept dimension
        concept_vectors = concept_vectors / (concept_vectors.norm(dim=-2, keepdim=True) + 1e-8)

    # Compute heatmaps via einsum
    heatmaps = einops.einsum(
        image_vectors,
        concept_vectors,
        "time layers batch patches dim, time layers batch concepts dim -> time layers batch concepts patches",
    )

    # Apply softmax across concepts
    if softmax:
        heatmaps = torch.nn.functional.softmax(heatmaps, dim=-2)

    # Select timesteps and layers
    heatmaps = heatmaps[timestep_indices]
    heatmaps = heatmaps[:, layer_indices]

    # Average over time and layers
    heatmaps = einops.reduce(
        heatmaps,
        "time layers batch concepts patches -> batch concepts patches",
        reduction="mean"
    )

    # Reshape to spatial grid
    heatmaps = einops.rearrange(
        heatmaps,
        "batch concepts (h w) -> batch concepts h w",
        h=height_tokens,
        w=width_tokens
    )

    return heatmaps


def heatmaps_to_pil_images(
    heatmaps: torch.Tensor,
    width: int,
    height: int,
    cmap: str = "plasma",
    normalize_globally: bool = True,
) -> List[PIL.Image.Image]:
    """
    Convert tensor heatmaps to colored PIL images.

    Args:
        heatmaps: Tensor of shape (num_concepts, h, w) or (batch, num_concepts, h, w).
        width: Target output width in pixels.
        height: Target output height in pixels.
        cmap: Matplotlib colormap name.
        normalize_globally: If True, normalize across all concepts. If False,
            normalize each concept independently.

    Returns:
        List of PIL images, one per concept.
    """
    # Handle batch dimension
    if len(heatmaps.shape) == 4:
        heatmaps = heatmaps[0]  # Take first batch

    heatmaps_np = heatmaps.cpu().float().numpy()

    if normalize_globally:
        global_min = heatmaps_np.min()
        global_max = heatmaps_np.max()
    else:
        global_min = None
        global_max = None

    colormap = plt.get_cmap(cmap)
    pil_images = []

    for concept_heatmap in heatmaps_np:
        # Normalize to [0, 1]
        if normalize_globally:
            normalized = (concept_heatmap - global_min) / (global_max - global_min + 1e-8)
        else:
            local_min = concept_heatmap.min()
            local_max = concept_heatmap.max()
            normalized = (concept_heatmap - local_min) / (local_max - local_min + 1e-8)

        # Apply colormap
        colored = colormap(normalized)
        rgb_image = (colored[:, :, :3] * 255).astype(np.uint8)
        pil_img = PIL.Image.fromarray(rgb_image)

        # Resize to target dimensions
        pil_img = pil_img.resize((width, height), resample=PIL.Image.NEAREST)
        pil_images.append(pil_img)

    return pil_images


def heatmaps_to_numpy(
    heatmaps: torch.Tensor,
    normalize: bool = True,
) -> np.ndarray:
    """
    Convert tensor heatmaps to numpy arrays.

    Args:
        heatmaps: Tensor of shape (num_concepts, h, w) or (batch, num_concepts, h, w).
        normalize: Whether to normalize to [0, 1].

    Returns:
        Numpy array with the same shape as input.
    """
    heatmaps_np = heatmaps.cpu().float().numpy()

    if normalize:
        global_min = heatmaps_np.min()
        global_max = heatmaps_np.max()
        heatmaps_np = (heatmaps_np - global_min) / (global_max - global_min + 1e-8)

    return heatmaps_np
