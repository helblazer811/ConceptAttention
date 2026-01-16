"""
Test script for concept encoders.

This module provides two encoder implementations:
1. PipelineConceptEncoder - Built on top of ConceptAttentionFluxPipeline.encode_image()
2. ProjectedConceptAttentionEncoder - Direct implementation with projection matrices

Usage:
    python test_concept_encoding.py
"""

import torch
import torch.nn as nn
import PIL
from typing import Union
import os

from .pipeline_encoder_flux import PipelineConceptEncoder

# ============ Test Functions ============


def test_pipeline_encoder():
    """Test the PipelineConceptEncoder."""
    print("=" * 60)
    print("Testing PipelineConceptEncoder")
    print("=" * 60)

    device = "cuda:0"
    model_name = "flux-schnell"

    # Initialize encoder
    print(f"\n1. Initializing PipelineConceptEncoder with model: {model_name}")
    encoder = PipelineConceptEncoder(
        model_name=model_name,
        device=device,
        offload=False,
    )
    print("   Encoder initialized successfully")

    # Create test image
    print("\n2. Creating test image...")
    test_image = PIL.Image.new("RGB", (512, 512), color="blue")
    pixels = test_image.load()
    for i in range(512):
        for j in range(512):
            pixels[i, j] = (i % 256, j % 256, (i + j) % 256)
    print(f"   Test image size: {test_image.size}")

    # Define concepts
    concepts = ["sky", "grass", "tree", "water"]
    print(f"\n3. Testing with concepts: {concepts}")

    # Run encode
    print("\n4. Running encode()...")
    concept_vectors, image_vectors = encoder.encode(
        image=test_image,
        concepts=concepts,
        layer_indices=[17],
        num_steps=4,
        noise_timestep=2,
    )

    # Print results
    print("\n5. Results:")
    if concept_vectors is not None:
        print(f"   Concept vectors shape: {concept_vectors.shape}")
        print(f"   Concept vectors dtype: {concept_vectors.dtype}")
    else:
        print("   Concept vectors: None")

    if image_vectors is not None:
        print(f"   Image vectors shape: {image_vectors.shape}")
        print(f"   Image vectors dtype: {image_vectors.dtype}")
    else:
        print("   Image vectors: None")

    # Test concept encoding
    print("\n6. Testing encode_concepts_to_input_space()...")
    concept_embeddings, concept_ids, vec = encoder.encode_concepts_to_input_space(
        concepts
    )
    print(f"   Concept embeddings shape: {concept_embeddings.shape}")
    print(f"   Concept IDs shape: {concept_ids.shape}")
    print(f"   Vec shape: {vec.shape}")

    print("\n" + "=" * 60)
    print("PipelineConceptEncoder test completed!")
    print("=" * 60)

    return encoder, concept_vectors, image_vectors


def test_with_real_image():
    """Test with a real image if available."""
    print("\n" + "=" * 60)
    print("Testing with real image")
    print("=" * 60)

    # Check for test images
    test_paths = [
        "results/cat.png",
        "results/image.png",
        "../test_concept_attention/results/cat.png",
    ]

    image_path = None
    for path in test_paths:
        if os.path.exists(path):
            image_path = path
            break

    if image_path is None:
        print("No test image found, skipping real image test")
        return None

    print(f"Using image: {image_path}")
    image = PIL.Image.open(image_path)

    device = "cuda:0"
    encoder = PipelineConceptEncoder(
        model_name="flux-schnell",
        device=device,
    )

    concepts = ["cat", "grass", "sky", "fur"]
    print(f"Concepts: {concepts}")

    concept_vectors, image_vectors = encoder.encode(
        image=image,
        concepts=concepts,
        layer_indices=[15, 16, 17, 18],
    )

    print(
        f"Concept vectors shape: {concept_vectors.shape if concept_vectors is not None else None}"
    )
    print(
        f"Image vectors shape: {image_vectors.shape if image_vectors is not None else None}"
    )

    return concept_vectors, image_vectors


if __name__ == "__main__":
    # Test pipeline encoder
    test_pipeline_encoder()

    # Test with real image if available
    test_with_real_image()
