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

    image, concept_attention_dict = generator.generate_image(
        width=1024,
        height=1024,
        num_steps=num_steps,
        guidance=0.0,
        seed=seed,
        prompt=prompt,
        concepts=concepts,
        joint_attention_kwargs=joint_attention_kwargs,
    )

    if target_space == "output":
        image_vecs = concept_attention_dict["output_space_image_vectors"]
        concept_vecs = concept_attention_dict["output_space_concept_vectors"]   
    elif target_space == "cross_attention":
        image_vecs = concept_attention_dict["cross_attention_image_vectors"]
        concept_vecs = concept_attention_dict["cross_attention_concept_vectors"]
    # Average over time 
    if average_over_time:
        image_vecs = image_vecs[average_after:].mean(dim=0)
        concept_vecs = concept_vecs[average_after:].mean(dim=0)
    if layer_index is not None:
        # Pull out the layer index
        concept_vectors = concept_vectors[layer_index]
        image_vectors = image_vectors[layer_index]

    # Apply linear normalization to concepts
    # NOTE: This is very important, as it makes up for not being able to do softmax 
    if normalize_concepts:
        concept_vectors = linear_normalization(concept_vectors, dim=-2)

    return image, concept_vectors, image_vectors