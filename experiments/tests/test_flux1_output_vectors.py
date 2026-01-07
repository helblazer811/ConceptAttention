"""
Tests for Flux1 pipeline output vector shapes.

These tests verify that concept_output_vectors and image_output_vectors
have the correct shapes based on the specified layer_indices and timesteps.
"""
import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch, MagicMock


class TestStackOutputVectorsFlux1:
    """Test the stack_output_vectors helper function for Flux1."""

    def test_stack_output_vectors_basic(self):
        """Test basic stacking with all layers and timesteps."""
        from concept_attention.utils import stack_output_vectors

        # Create mock attention dicts: 2 timesteps, 3 layers each
        batch_size = 1
        num_concepts = 2
        num_img_tokens = 64
        hidden_dim = 128

        concept_attention_dicts = []
        for t in range(2):  # 2 timesteps
            timestep_layers = []
            for layer in range(3):  # 3 layers
                layer_dict = {
                    "output_space_concept_vectors": torch.randn(batch_size, num_concepts, hidden_dim),
                    "output_space_image_vectors": torch.randn(batch_size, num_img_tokens, hidden_dim),
                    "concept_scores": torch.randn(batch_size, num_concepts, num_img_tokens),
                }
                timestep_layers.append(layer_dict)
            concept_attention_dicts.append(timestep_layers)

        # Stack concept vectors
        result = stack_output_vectors(concept_attention_dicts, "output_space_concept_vectors")

        # Expected shape: (batch, time, layers, tokens, dim)
        assert result.shape == (batch_size, 2, 3, num_concepts, hidden_dim)

    def test_stack_output_vectors_image(self):
        """Test stacking image output vectors."""
        from concept_attention.utils import stack_output_vectors

        batch_size = 1
        num_img_tokens = 256
        hidden_dim = 64

        concept_attention_dicts = []
        for t in range(3):  # 3 timesteps
            timestep_layers = []
            for layer in range(2):  # 2 layers
                layer_dict = {
                    "output_space_image_vectors": torch.randn(batch_size, num_img_tokens, hidden_dim),
                }
                timestep_layers.append(layer_dict)
            concept_attention_dicts.append(timestep_layers)

        result = stack_output_vectors(concept_attention_dicts, "output_space_image_vectors")

        # Expected shape: (batch, time, layers, tokens, dim)
        assert result.shape == (batch_size, 3, 2, num_img_tokens, hidden_dim)

    def test_stack_output_vectors_missing_key(self):
        """Test that missing keys are handled gracefully."""
        from concept_attention.utils import stack_output_vectors

        concept_attention_dicts = []
        for t in range(2):
            timestep_layers = []
            for layer in range(2):
                # Only include concept_scores, not vectors
                layer_dict = {"concept_scores": torch.randn(1, 2, 64)}
                timestep_layers.append(layer_dict)
            concept_attention_dicts.append(timestep_layers)

        result = stack_output_vectors(concept_attention_dicts, "output_space_concept_vectors")

        # Should return None when key is missing
        assert result is None

    def test_stack_output_vectors_partial_layers(self):
        """Test stacking when only some layers have vectors (selective caching)."""
        from concept_attention.utils import stack_output_vectors

        batch_size = 1
        num_concepts = 3
        hidden_dim = 64

        concept_attention_dicts = []
        for t in range(2):
            timestep_layers = []
            for layer in range(4):
                layer_dict = {"concept_scores": torch.randn(1, num_concepts, 64)}
                # Only layers 2 and 3 have vectors (simulating layer_indices=[2, 3])
                if layer in [2, 3]:
                    layer_dict["output_space_concept_vectors"] = torch.randn(batch_size, num_concepts, hidden_dim)
                timestep_layers.append(layer_dict)
            concept_attention_dicts.append(timestep_layers)

        result = stack_output_vectors(concept_attention_dicts, "output_space_concept_vectors")

        # Should only have 2 layers (the ones with vectors)
        assert result.shape == (batch_size, 2, 2, num_concepts, hidden_dim)


class TestConditionalVectorCachingLogicFlux1:
    """Test the conditional caching logic for Flux1."""

    def test_should_cache_logic_layer_in_indices(self):
        """Test should_cache evaluates True when layer is in indices."""
        cache_vectors = True
        layer_indices = [15, 16, 17, 18]
        current_layer_idx = 16

        should_cache = cache_vectors and (
            layer_indices is None or current_layer_idx in layer_indices
        )
        assert should_cache is True

    def test_should_cache_logic_layer_not_in_indices(self):
        """Test should_cache evaluates False when layer not in indices."""
        cache_vectors = True
        layer_indices = [15, 16, 17, 18]
        current_layer_idx = 0

        should_cache = cache_vectors and (
            layer_indices is None or current_layer_idx in layer_indices
        )
        assert should_cache is False

    def test_should_cache_logic_cache_vectors_false(self):
        """Test should_cache evaluates False when cache_vectors=False."""
        cache_vectors = False
        layer_indices = None  # Would normally cache all
        current_layer_idx = 0

        should_cache = cache_vectors and (
            layer_indices is None or current_layer_idx in layer_indices
        )
        assert should_cache is False

    def test_should_cache_logic_layer_indices_none(self):
        """Test should_cache evaluates True when layer_indices=None (cache all)."""
        cache_vectors = True
        layer_indices = None
        current_layer_idx = 10  # Any layer

        should_cache = cache_vectors and (
            layer_indices is None or current_layer_idx in layer_indices
        )
        assert should_cache is True


class TestOutputShapeCalculationsFlux1:
    """Test expected output shapes for Flux1 configurations."""

    def test_expected_shape_default_indices(self):
        """Test expected shape with default layer_indices=[15,16,17,18]."""
        layer_indices = [15, 16, 17, 18]
        num_layers = len(layer_indices)
        num_timesteps = 4  # Flux schnell default

        batch_size = 1
        num_concepts = 2
        # For 1024x1024 image: (1024/16) * (1024/16) = 64 * 64 = 4096 tokens
        num_img_tokens = 64 * 64
        hidden_dim = 3072  # Flux hidden size

        expected_concept_shape = (batch_size, num_timesteps, num_layers, num_concepts, hidden_dim)
        expected_image_shape = (batch_size, num_timesteps, num_layers, num_img_tokens, hidden_dim)

        assert expected_concept_shape == (1, 4, 4, 2, 3072)
        assert expected_image_shape == (1, 4, 4, 4096, 3072)

    def test_expected_shape_custom_indices(self):
        """Test expected shape with custom layer and timestep indices."""
        layer_indices = [10, 12, 14, 16, 18]  # 5 layers
        timestep_indices = [0, 1, 2]  # 3 timesteps

        batch_size = 1
        num_concepts = 5
        num_img_tokens = 64 * 64  # 1024x1024 image
        hidden_dim = 3072

        expected_concept_shape = (batch_size, len(timestep_indices), len(layer_indices), num_concepts, hidden_dim)
        expected_image_shape = (batch_size, len(timestep_indices), len(layer_indices), num_img_tokens, hidden_dim)

        assert expected_concept_shape == (1, 3, 5, 5, 3072)
        assert expected_image_shape == (1, 3, 5, 4096, 3072)


class TestPipelineOutputClassFlux1:
    """Test the ConceptAttentionPipelineOutput dataclass for Flux1."""

    def test_output_class_has_vector_fields(self):
        """Test that output class has the new vector fields."""
        from concept_attention.flux.pipeline import ConceptAttentionPipelineOutput
        import PIL.Image

        # Create a minimal output
        dummy_image = PIL.Image.new("RGB", (64, 64))
        output = ConceptAttentionPipelineOutput(
            image=dummy_image,
            concept_heatmaps=[],
            cross_attention_maps=[],
            concept_output_vectors=torch.randn(1, 2, 3, 4, 64),
            image_output_vectors=torch.randn(1, 2, 3, 100, 64),
        )

        assert output.concept_output_vectors is not None
        assert output.image_output_vectors is not None
        assert output.concept_output_vectors.shape == (1, 2, 3, 4, 64)
        assert output.image_output_vectors.shape == (1, 2, 3, 100, 64)

    def test_output_class_vectors_default_none(self):
        """Test that vector fields default to None."""
        from concept_attention.flux.pipeline import ConceptAttentionPipelineOutput
        import PIL.Image

        dummy_image = PIL.Image.new("RGB", (64, 64))
        output = ConceptAttentionPipelineOutput(
            image=dummy_image,
            concept_heatmaps=[],
            cross_attention_maps=[],
        )

        assert output.concept_output_vectors is None
        assert output.image_output_vectors is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
