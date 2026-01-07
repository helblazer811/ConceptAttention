"""
Shared attention score computation utilities for all concept attention pipelines.
"""
import torch
import einops


def compute_concept_attention_scores(
    concept_outputs: torch.Tensor,
    image_outputs: torch.Tensor,
    normalize: bool = False,
) -> torch.Tensor:
    """
    Compute attention scores between concept and image outputs (output space attention).

    This computes the dot product similarity between concept representations
    and image patch representations in the output space of attention layers.

    Args:
        concept_outputs: Concept representations, shape (batch, num_concepts, dim).
        image_outputs: Image patch representations, shape (batch, num_img_tokens, dim).
        normalize: Whether to L2-normalize before computing scores.

    Returns:
        Attention scores of shape (batch, num_concepts, num_img_tokens).
    """
    if normalize:
        concept_outputs = torch.nn.functional.normalize(concept_outputs, dim=-1)
        image_outputs = torch.nn.functional.normalize(image_outputs, dim=-1)

    scores = einops.einsum(
        concept_outputs,
        image_outputs,
        "batch num_concepts dim, batch num_img dim -> batch num_concepts num_img"
    )

    return scores


def compute_cross_attention_scores(
    concept_q: torch.Tensor,
    image_k: torch.Tensor,
    head_dim: int = None,
) -> torch.Tensor:
    """
    Compute Q·K cross-attention scores between concepts and image patches.

    This computes attention scores where concepts are queries attending to
    image patches as keys, giving a measure of how much each concept
    "looks at" each image region.

    Args:
        concept_q: Concept queries, shape (batch, heads, num_concepts, head_dim)
            or (batch, num_concepts, dim).
        image_k: Image keys, shape (batch, heads, num_img_tokens, head_dim)
            or (batch, num_img_tokens, dim).
        head_dim: If provided, used for scaling. Otherwise inferred from tensors.

    Returns:
        Attention scores of shape (batch, num_concepts, num_img_tokens).
        If inputs have head dimension, heads are averaged.
    """
    has_heads = len(concept_q.shape) == 4

    if has_heads:
        # Shape: (batch, heads, num_concepts, head_dim) x (batch, heads, num_img, head_dim)
        # -> (batch, heads, num_concepts, num_img)
        scores = einops.einsum(
            concept_q,
            image_k,
            "batch heads num_concepts dim, batch heads num_img dim -> batch heads num_concepts num_img",
        )
        # Average over heads
        scores = einops.reduce(
            scores,
            "batch heads num_concepts num_img -> batch num_concepts num_img",
            "mean"
        )
        if head_dim is None:
            head_dim = concept_q.shape[-1]
    else:
        # Shape: (batch, num_concepts, dim) x (batch, num_img, dim)
        # -> (batch, num_concepts, num_img)
        scores = einops.einsum(
            concept_q,
            image_k,
            "batch num_concepts dim, batch num_img dim -> batch num_concepts num_img",
        )
        if head_dim is None:
            head_dim = concept_q.shape[-1]

    # Apply scaling (standard attention scaling)
    scores = scores / (head_dim ** 0.5)

    return scores


def compute_bidirectional_attention_scores(
    concept_outputs: torch.Tensor,
    image_outputs: torch.Tensor,
    concept_q: torch.Tensor = None,
    concept_k: torch.Tensor = None,
    image_q: torch.Tensor = None,
    image_k: torch.Tensor = None,
) -> dict:
    """
    Compute both output-space and cross-attention scores.

    This is a convenience function that computes multiple types of attention
    scores commonly used in concept attention analysis.

    Args:
        concept_outputs: Concept representations in output space.
        image_outputs: Image representations in output space.
        concept_q: Concept query vectors (optional, for cross-attention).
        concept_k: Concept key vectors (optional, for cross-attention).
        image_q: Image query vectors (optional, for cross-attention).
        image_k: Image key vectors (optional, for cross-attention).

    Returns:
        Dictionary with computed attention scores:
        - "concept_scores": Output-space attention (concept -> image)
        - "cross_attention_scores": Q·K attention (concept_q -> image_k)
        - "reverse_cross_attention_scores": Q·K attention (image_q -> concept_k)
    """
    result = {}

    # Output space attention
    result["concept_scores"] = compute_concept_attention_scores(
        concept_outputs, image_outputs
    )

    # Cross-attention: concepts attending to images
    if concept_q is not None and image_k is not None:
        result["cross_attention_scores"] = compute_cross_attention_scores(
            concept_q, image_k
        )

    # Reverse cross-attention: images attending to concepts
    if image_q is not None and concept_k is not None:
        result["reverse_cross_attention_scores"] = compute_cross_attention_scores(
            image_q, concept_k
        )

    return result


def aggregate_attention_heads(
    attention_scores: torch.Tensor,
    method: str = "mean",
) -> torch.Tensor:
    """
    Aggregate attention scores across multiple heads.

    Args:
        attention_scores: Scores with head dimension,
            shape (batch, heads, seq1, seq2).
        method: Aggregation method - "mean", "max", or "sum".

    Returns:
        Aggregated scores of shape (batch, seq1, seq2).
    """
    if method == "mean":
        return attention_scores.mean(dim=1)
    elif method == "max":
        return attention_scores.max(dim=1)[0]
    elif method == "sum":
        return attention_scores.sum(dim=1)
    else:
        raise ValueError(f"Unknown aggregation method: {method}")
