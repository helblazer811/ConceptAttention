"""
Test generating concept heatmaps for a generated image.
"""
import pytest
import os


def test_generate_image_with_concepts(pipeline, tmp_path):
    """Test generating an image with concept heatmaps."""
    prompt = "A dragon in the forest."
    concepts = ["dragon", "forest", "fire", "scale", "claw"]

    pipeline_output = pipeline.generate_image(
        prompt=prompt,
        concepts=concepts,
        width=1024,
        height=1024
    )

    # Verify outputs exist
    assert pipeline_output.image is not None
    assert pipeline_output.concept_heatmaps is not None
    assert len(pipeline_output.concept_heatmaps) == len(concepts)

    # Save outputs to temp directory
    image = pipeline_output.image
    concept_heatmaps = pipeline_output.concept_heatmaps

    image.save(tmp_path / "image.png")
    for concept, concept_heatmap in zip(concepts, concept_heatmaps):
        concept_heatmap.save(tmp_path / f"{concept}.png")

    # Verify files were saved
    assert (tmp_path / "image.png").exists()
    for concept in concepts:
        assert (tmp_path / f"{concept}.png").exists()
