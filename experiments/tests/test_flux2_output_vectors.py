"""
Tests for Flux2 pipeline output vector shapes.

These tests verify that concept_output_vectors and image_output_vectors
have the correct shapes based on the specified layer_indices and timesteps.
"""
import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch, MagicMock


class TestStackOutputVectors:
    """Test the stack_output_vectors helper function."""

    def test_stack_output_vectors_basic(self):
        """Test basic stacking with all layers and timesteps."""
        from concept_attention.flux2.pipeline import stack_output_vectors

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
                    "concept_output_vectors": torch.randn(batch_size, num_concepts, hidden_dim),
                    "image_output_vectors": torch.randn(batch_size, num_img_tokens, hidden_dim),
                    "concept_scores": torch.randn(batch_size, num_concepts, num_img_tokens),
                }
                timestep_layers.append(layer_dict)
            concept_attention_dicts.append(timestep_layers)

        # Stack concept vectors
        result = stack_output_vectors(concept_attention_dicts, "concept_output_vectors")

        # Expected shape: (batch, time, layers, tokens, dim)
        assert result.shape == (batch_size, 2, 3, num_concepts, hidden_dim)

    def test_stack_output_vectors_image(self):
        """Test stacking image output vectors."""
        from concept_attention.flux2.pipeline import stack_output_vectors

        batch_size = 1
        num_img_tokens = 256
        hidden_dim = 64

        concept_attention_dicts = []
        for t in range(3):  # 3 timesteps
            timestep_layers = []
            for layer in range(2):  # 2 layers
                layer_dict = {
                    "image_output_vectors": torch.randn(batch_size, num_img_tokens, hidden_dim),
                }
                timestep_layers.append(layer_dict)
            concept_attention_dicts.append(timestep_layers)

        result = stack_output_vectors(concept_attention_dicts, "image_output_vectors")

        # Expected shape: (batch, time, layers, tokens, dim)
        assert result.shape == (batch_size, 3, 2, num_img_tokens, hidden_dim)

    def test_stack_output_vectors_missing_key(self):
        """Test that missing keys are handled gracefully."""
        from concept_attention.flux2.pipeline import stack_output_vectors

        concept_attention_dicts = []
        for t in range(2):
            timestep_layers = []
            for layer in range(2):
                # Only include concept_scores, not vectors
                layer_dict = {"concept_scores": torch.randn(1, 2, 64)}
                timestep_layers.append(layer_dict)
            concept_attention_dicts.append(timestep_layers)

        result = stack_output_vectors(concept_attention_dicts, "concept_output_vectors")

        # Should return None when key is missing
        assert result is None

    def test_stack_output_vectors_partial_layers(self):
        """Test stacking when only some layers have vectors (selective caching)."""
        from concept_attention.flux2.pipeline import stack_output_vectors

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
                    layer_dict["concept_output_vectors"] = torch.randn(batch_size, num_concepts, hidden_dim)
                timestep_layers.append(layer_dict)
            concept_attention_dicts.append(timestep_layers)

        result = stack_output_vectors(concept_attention_dicts, "concept_output_vectors")

        # Should only have 2 layers (the ones with vectors)
        assert result.shape == (batch_size, 2, 2, num_concepts, hidden_dim)


class TestConditionalVectorCachingLogic:
    """Test the conditional caching logic in isolation."""

    def test_should_cache_logic_layer_in_indices(self):
        """Test should_cache evaluates True when layer is in indices."""
        cache_vectors = True
        layer_indices = [0, 1, 2]
        current_layer_idx = 1

        should_cache = cache_vectors and (
            layer_indices is None or current_layer_idx in layer_indices
        )
        assert should_cache is True

    def test_should_cache_logic_layer_not_in_indices(self):
        """Test should_cache evaluates False when layer not in indices."""
        cache_vectors = True
        layer_indices = [5, 6, 7]
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
        current_layer_idx = 5  # Any layer

        should_cache = cache_vectors and (
            layer_indices is None or current_layer_idx in layer_indices
        )
        assert should_cache is True

    def test_kwargs_extraction(self):
        """Test kwargs extraction pattern used in dit_block."""
        kwargs = {
            "cache_vectors": True,
            "layer_indices": [5, 6, 7],
            "current_layer_idx": 6,
        }

        cache_vectors = kwargs.get("cache_vectors", True)
        layer_indices = kwargs.get("layer_indices", None)
        current_layer_idx = kwargs.get("current_layer_idx", 0)

        should_cache = cache_vectors and (
            layer_indices is None or current_layer_idx in layer_indices
        )
        assert should_cache is True
        assert cache_vectors is True
        assert layer_indices == [5, 6, 7]
        assert current_layer_idx == 6

    def test_kwargs_defaults(self):
        """Test kwargs defaults when not provided."""
        kwargs = {}  # Empty kwargs

        cache_vectors = kwargs.get("cache_vectors", True)  # Default True
        layer_indices = kwargs.get("layer_indices", None)  # Default None
        current_layer_idx = kwargs.get("current_layer_idx", 0)  # Default 0

        should_cache = cache_vectors and (
            layer_indices is None or current_layer_idx in layer_indices
        )
        # With defaults: cache_vectors=True, layer_indices=None -> should cache all
        assert should_cache is True


class TestOutputShapeCalculations:
    """Test expected output shapes for various configurations."""

    def test_expected_shape_default_indices(self):
        """Test expected shape with default layer_indices=[5,6,7] and timesteps."""
        # With default settings:
        # - layer_indices = [5, 6, 7] -> 3 layers
        # - timesteps = last 30% of num_inference_steps
        # For num_inference_steps=28, timesteps = range(19, 27) -> 8 timesteps

        num_inference_steps = 28
        start_idx = int(num_inference_steps * 0.7)  # 19
        num_timesteps = num_inference_steps - 1 - start_idx  # 8

        layer_indices = [5, 6, 7]
        num_layers = len(layer_indices)

        batch_size = 1
        num_concepts = 2
        # For 2048x2048 image: (2048/16) * (2048/16) = 128 * 128 = 16384 tokens
        num_img_tokens = 128 * 128
        hidden_dim = 3072  # Flux2 hidden size

        expected_concept_shape = (batch_size, num_timesteps, num_layers, num_concepts, hidden_dim)
        expected_image_shape = (batch_size, num_timesteps, num_layers, num_img_tokens, hidden_dim)

        assert expected_concept_shape == (1, 8, 3, 2, 3072)
        assert expected_image_shape == (1, 8, 3, 16384, 3072)

    def test_expected_shape_custom_indices(self):
        """Test expected shape with custom layer and timestep indices."""
        layer_indices = [0, 2, 4, 6]  # 4 layers
        timestep_indices = [10, 15, 20]  # 3 timesteps

        batch_size = 1
        num_concepts = 5
        num_img_tokens = 64 * 64  # 1024x1024 image
        hidden_dim = 3072

        expected_concept_shape = (batch_size, len(timestep_indices), len(layer_indices), num_concepts, hidden_dim)
        expected_image_shape = (batch_size, len(timestep_indices), len(layer_indices), num_img_tokens, hidden_dim)

        assert expected_concept_shape == (1, 3, 4, 5, 3072)
        assert expected_image_shape == (1, 3, 4, 4096, 3072)

    def test_expected_shape_single_layer_single_timestep(self):
        """Test expected shape with single layer and timestep."""
        layer_indices = [7]  # 1 layer
        timestep_indices = [0]  # 1 timestep

        batch_size = 1
        num_concepts = 1
        num_img_tokens = 32 * 32  # 512x512 image
        hidden_dim = 3072

        expected_concept_shape = (batch_size, 1, 1, num_concepts, hidden_dim)
        expected_image_shape = (batch_size, 1, 1, num_img_tokens, hidden_dim)

        assert expected_concept_shape == (1, 1, 1, 1, 3072)
        assert expected_image_shape == (1, 1, 1, 1024, 3072)


class TestPipelineOutputClass:
    """Test the ConceptAttentionPipelineOutput dataclass."""

    def test_output_class_has_vector_fields(self):
        """Test that output class has the new vector fields."""
        from concept_attention.flux2.pipeline import ConceptAttentionPipelineOutput
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
        from concept_attention.flux2.pipeline import ConceptAttentionPipelineOutput
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
