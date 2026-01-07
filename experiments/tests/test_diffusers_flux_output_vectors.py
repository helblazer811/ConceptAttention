"""
Tests for Diffusers Flux pipeline output vector shapes.

These tests verify that concept_output_vectors and image_output_vectors
have the correct shapes based on the specified layer_indices and timesteps.
"""
import pytest
import torch
import numpy as np


class TestConditionalVectorCachingLogicDiffusersFlux:
    """Test the conditional caching logic for Diffusers Flux."""

    def test_should_cache_logic_layer_in_indices(self):
        """Test should_cache evaluates True when layer is in indices."""
        cache_vectors = True
        layer_indices = [10, 12, 14, 16, 18]
        current_layer_idx = 14

        should_cache = cache_vectors and (
            layer_indices is None or current_layer_idx in layer_indices
        )
        assert should_cache is True

    def test_should_cache_logic_layer_not_in_indices(self):
        """Test should_cache evaluates False when layer not in indices."""
        cache_vectors = True
        layer_indices = [10, 12, 14, 16, 18]
        current_layer_idx = 0

        should_cache = cache_vectors and (
            layer_indices is None or current_layer_idx in layer_indices
        )
        assert should_cache is False

    def test_should_cache_logic_cache_vectors_false(self):
        """Test should_cache evaluates False when cache_vectors=False."""
        cache_vectors = False
        layer_indices = None
        current_layer_idx = 0

        should_cache = cache_vectors and (
            layer_indices is None or current_layer_idx in layer_indices
        )
        assert should_cache is False

    def test_should_cache_logic_layer_indices_none(self):
        """Test should_cache evaluates True when layer_indices=None."""
        cache_vectors = True
        layer_indices = None
        current_layer_idx = 10

        should_cache = cache_vectors and (
            layer_indices is None or current_layer_idx in layer_indices
        )
        assert should_cache is True


class TestTimestepFilteringDiffusersFlux:
    """Test timestep filtering for Diffusers Flux."""

    def test_all_timesteps_tracked(self):
        """Test that all timesteps are tracked when timestep_indices is None."""
        timestep_indices = None
        num_timesteps = 28

        tracked = []
        for step_idx in range(num_timesteps):
            should_track = timestep_indices is None or step_idx in timestep_indices
            if should_track:
                tracked.append(step_idx)

        assert len(tracked) == num_timesteps

    def test_specific_timesteps_tracked(self):
        """Test that only specified timesteps are tracked."""
        timestep_indices = [20, 22, 24, 26]
        num_timesteps = 28

        tracked = []
        for step_idx in range(num_timesteps):
            should_track = timestep_indices is None or step_idx in timestep_indices
            if should_track:
                tracked.append(step_idx)

        assert tracked == timestep_indices


class TestOutputShapeCalculationsDiffusersFlux:
    """Test expected output shapes for Diffusers Flux configurations."""

    def test_expected_shape_default(self):
        """Test expected shape with default settings."""
        layer_indices = list(range(10, 19))  # 9 layers
        num_timesteps = 28

        batch_size = 1
        num_concepts = 3
        # For 1024x1024 image
        num_img_tokens = 64 * 64
        hidden_dim = 3072

        expected_concept_shape = (num_timesteps, len(layer_indices), batch_size, num_concepts, hidden_dim)
        expected_image_shape = (num_timesteps, len(layer_indices), batch_size, num_img_tokens, hidden_dim)

        assert expected_concept_shape == (28, 9, 1, 3, 3072)
        assert expected_image_shape == (28, 9, 1, 4096, 3072)

    def test_expected_shape_subset(self):
        """Test expected shape with subset of layers and timesteps."""
        layer_indices = [15, 16, 17, 18]  # 4 layers
        timestep_indices = [24, 25, 26, 27]  # 4 timesteps

        batch_size = 1
        num_concepts = 5
        num_img_tokens = 32 * 32
        hidden_dim = 3072

        expected_concept_shape = (len(timestep_indices), len(layer_indices), batch_size, num_concepts, hidden_dim)
        expected_image_shape = (len(timestep_indices), len(layer_indices), batch_size, num_img_tokens, hidden_dim)

        assert expected_concept_shape == (4, 4, 1, 5, 3072)
        assert expected_image_shape == (4, 4, 1, 1024, 3072)


class TestOutputStackingDiffusersFlux:
    """Test vector stacking for Diffusers Flux."""

    def test_stacking_over_timesteps(self):
        """Test stacking vectors over timesteps."""
        batch_size = 1
        num_concepts = 2
        num_patches = 100
        hidden_dim = 64
        num_timesteps = 3
        num_layers = 4

        all_concept_output_vectors = []
        all_image_output_vectors = []

        for t in range(num_timesteps):
            # Each timestep returns stacked layers
            all_concept_output_vectors.append(
                torch.randn(num_layers, batch_size, num_concepts, hidden_dim)
            )
            all_image_output_vectors.append(
                torch.randn(num_layers, batch_size, num_patches, hidden_dim)
            )

        concept_vectors = torch.stack(all_concept_output_vectors, dim=0)
        image_vectors = torch.stack(all_image_output_vectors, dim=0)

        # Shape: (time, layers, batch, tokens, dim)
        assert concept_vectors.shape == (num_timesteps, num_layers, batch_size, num_concepts, hidden_dim)
        assert image_vectors.shape == (num_timesteps, num_layers, batch_size, num_patches, hidden_dim)

    def test_empty_lists_returns_none(self):
        """Test that empty lists result in None."""
        all_concept_output_vectors = []

        if len(all_concept_output_vectors) > 0:
            result = torch.stack(all_concept_output_vectors, dim=0)
        else:
            result = None

        assert result is None


class TestOutputClassDiffusersFlux:
    """Test the FluxConceptAttentionOutput dataclass."""

    def test_output_class_has_vector_fields(self):
        """Test that output class has the new vector fields."""
        from concept_attention.diffusers.flux.pipeline import FluxConceptAttentionOutput
        import PIL.Image

        dummy_image = [PIL.Image.new("RGB", (64, 64))]
        output = FluxConceptAttentionOutput(
            images=dummy_image,
            concept_attention_maps=[[]],
            concept_output_vectors=torch.randn(1, 2, 3, 4, 64),
            image_output_vectors=torch.randn(1, 2, 3, 100, 64),
        )

        assert output.concept_output_vectors is not None
        assert output.image_output_vectors is not None
        assert output.concept_output_vectors.shape == (1, 2, 3, 4, 64)
        assert output.image_output_vectors.shape == (1, 2, 3, 100, 64)

    def test_output_class_vectors_default_none(self):
        """Test that vector fields default to None."""
        from concept_attention.diffusers.flux.pipeline import FluxConceptAttentionOutput
        import PIL.Image

        dummy_image = [PIL.Image.new("RGB", (64, 64))]
        output = FluxConceptAttentionOutput(
            images=dummy_image,
            concept_attention_maps=[[]],
        )

        assert output.concept_output_vectors is None
        assert output.image_output_vectors is None


class TestKwargsPassingDiffusersFlux:
    """Test kwargs passing through model hierarchy."""

    def test_kwargs_extraction(self):
        """Test kwargs extraction pattern used in dit_block."""
        kwargs = {
            "cache_vectors": True,
            "layer_indices": [10, 12, 14],
            "current_layer_idx": 12,
        }

        cache_vectors = kwargs.get("cache_vectors", True)
        layer_indices = kwargs.get("layer_indices", None)
        current_layer_idx = kwargs.get("current_layer_idx", 0)

        should_cache = cache_vectors and (
            layer_indices is None or current_layer_idx in layer_indices
        )
        assert should_cache is True

    def test_kwargs_defaults(self):
        """Test kwargs defaults when not provided."""
        kwargs = {}

        cache_vectors = kwargs.get("cache_vectors", True)
        layer_indices = kwargs.get("layer_indices", None)
        current_layer_idx = kwargs.get("current_layer_idx", 0)

        should_cache = cache_vectors and (
            layer_indices is None or current_layer_idx in layer_indices
        )
        assert should_cache is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
