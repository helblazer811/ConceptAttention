import torch
import einops 

from concept_attention.utils import linear_normalization
from concept_attention.image_generator import FluxGenerator

def generate_concept_basis_and_image_queries(
    prompt: str,
    concepts: list[str],
    layer_index: list[int] = [18],
    average_over_time: bool=True,
    model_name="flux-dev",
    num_steps=50,
    seed=42,
    average_after=0,
    target_space="output",
    generator=None,
    normalize_concepts=False,
    device="cuda",
    include_images_in_basis=False,
    offload=True,
    joint_attention_kwargs=None
):
    """
        Given a prompt, generate the set basis of concept vectors
        for a particular layer in the model and the encoded image queries.
    """
    assert target_space in ["output", "value", "cross_attention"], "Invalid target space"
    if generator is None:
        generator = FluxGenerator(
            model_name,
            device, 
            offload=offload,
        )

    image = generator.generate_image(
        width=1024,
        height=1024,
        num_steps=num_steps,
        guidance=0.0,
        seed=seed,
        prompt=prompt,
        concepts=concepts,
        joint_attention_kwargs=joint_attention_kwargs,
    )

    concept_vectors = []
    image_vectors = []
    supplemental_vectors = []
    for double_block in generator.model.double_blocks:
        if target_space == "output":
            image_vecs = torch.stack(
                double_block.image_output_vectors
            ).squeeze(1)
            concept_vecs = torch.stack(
                double_block.concept_output_vectors
            ).squeeze(1)
            image_supplemental_vecs = image_vecs
            # Clear out the layer
            double_block.clear_cached_vectors()
        elif target_space == "value":
            image_vecs = torch.stack(
                double_block.image_value_vectors
            ).squeeze(1)
            concept_vecs = torch.stack(
                double_block.concept_value_vectors
            ).squeeze(1)
            image_supplemental_vecs = image_vecs
            # Clear out the layer
            double_block.clear_cached_vectors()
        elif target_space == "cross_attention":
            image_vecs = torch.stack(
                double_block.image_query_vectors
            ).squeeze(1)
            concept_vecs = torch.stack(
                double_block.concept_key_vectors
            ).squeeze(1)
            image_supplemental_vecs = torch.stack(
                double_block.image_key_vectors
            ).squeeze(1)
            # Clear out the layer
            double_block.clear_cached_vectors()
        else:
            raise ValueError("Invalid target space")
        # Average over time 
        if average_over_time:
            image_vecs = image_vecs[average_after:].mean(dim=0)
            concept_vecs = concept_vecs[average_after:].mean(dim=0)
            image_supplemental_vecs = image_supplemental_vecs[average_after:].mean(dim=0)
        # Add to list
        concept_vectors.append(concept_vecs)
        image_vectors.append(image_vecs)
        supplemental_vectors.append(image_supplemental_vecs)
    # Stack layers
    concept_vectors = torch.stack(concept_vectors)
    if include_images_in_basis:
        supplemental_vectors = torch.stack(supplemental_vectors)
        concept_vectors = torch.cat([concept_vectors, supplemental_vectors], dim=-2)
    image_vectors = torch.stack(image_vectors)

    if layer_index is not None:
        # Pull out the layer index
        concept_vectors = concept_vectors[layer_index]
        image_vectors = image_vectors[layer_index]

    # Apply linear normalization to concepts
    # NOTE: This is very important, as it makes up for not being able to do softmax 
    if normalize_concepts:
        concept_vectors = linear_normalization(concept_vectors, dim=-2)

    return image, concept_vectors, image_vectors