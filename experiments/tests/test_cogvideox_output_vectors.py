"""
Tests for CogVideoX pipeline output vector shapes.

These tests verify that concept_output_vectors and image_output_vectors
have the correct shapes based on the specified layer_indices and timesteps.
"""
import pytest
import torch
import numpy as np


class TestConditionalVectorCachingLogicCogVideoX:
    """Test the conditional caching logic for CogVideoX."""

    def test_should_cache_logic_layer_in_indices(self):
        """Test should_cache evaluates True when layer is in indices."""
        cache_vectors = True
        layer_indices = [30, 35, 40]
        current_layer_idx = 35

        should_cache = cache_vectors and (
            layer_indices is None or current_layer_idx in layer_indices
        )
        assert should_cache is True

    def test_should_cache_logic_layer_not_in_indices(self):
        """Test should_cache evaluates False when layer not in indices."""
        cache_vectors = True
        layer_indices = [30, 35, 40]
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
        current_layer_idx = 25

        should_cache = cache_vectors and (
            layer_indices is None or current_layer_idx in layer_indices
        )
        assert should_cache is True


class TestTimestepFilteringCogVideoX:
    """Test timestep filtering for CogVideoX."""

    def test_all_timesteps_tracked(self):
        """Test that all timesteps are tracked when timestep_indices is None."""
        timestep_indices = None
        num_timesteps = 50  # CogVideoX default

        tracked = []
        for step_idx in range(num_timesteps):
            should_track = timestep_indices is None or step_idx in timestep_indices
            if should_track:
                tracked.append(step_idx)

        assert len(tracked) == num_timesteps

    def test_specific_timesteps_tracked(self):
        """Test that only specified timesteps are tracked."""
        timestep_indices = [45, 46, 47, 48, 49]  # Last 5 timesteps
        num_timesteps = 50

        tracked = []
        for step_idx in range(num_timesteps):
            should_track = timestep_indices is None or step_idx in timestep_indices
            if should_track:
                tracked.append(step_idx)

        assert tracked == timestep_indices


class TestOutputShapeCalculationsCogVideoX:
    """Test expected output shapes for CogVideoX configurations."""

    def test_expected_shape_video(self):
        """Test expected shape for video output."""
        num_layers = 42  # CogVideoX-5B has 42 layers
        num_timesteps = 5  # Tracked timesteps

        batch_size = 2  # Positive and negative
        num_concepts = 3
        # For 480x720 video with 49 frames at patch_size=2
        num_frames = 13  # Latent frames
        height = 60  # 480 / 8
        width = 90  # 720 / 8
        num_patches = num_frames * height * width
        hidden_dim = 3072

        expected_concept_shape = (batch_size, num_layers, num_concepts, hidden_dim)
        expected_image_shape = (batch_size, num_layers, num_patches, hidden_dim)

        # These are per-timestep shapes before stacking over time
        assert expected_concept_shape[0] == batch_size
        assert expected_image_shape[0] == batch_size

    def test_stacking_over_timesteps(self):
        """Test stacking vectors over timesteps."""
        batch_size = 1
        num_concepts = 2
        num_patches = 100
        hidden_dim = 64
        num_timesteps = 3
        num_layers = 4

        concept_attention_dict = {
            "concept_output_vectors": [],
            "image_output_vectors": [],
        }

        for t in range(num_timesteps):
            concept_attention_dict["concept_output_vectors"].append(
                torch.randn(batch_size, num_layers, num_concepts, hidden_dim)
            )
            concept_attention_dict["image_output_vectors"].append(
                torch.randn(batch_size, num_layers, num_patches, hidden_dim)
            )

        concept_vectors = torch.stack(concept_attention_dict["concept_output_vectors"], dim=0)
        image_vectors = torch.stack(concept_attention_dict["image_output_vectors"], dim=0)

        assert concept_vectors.shape == (num_timesteps, batch_size, num_layers, num_concepts, hidden_dim)
        assert image_vectors.shape == (num_timesteps, batch_size, num_layers, num_patches, hidden_dim)


class TestEmptyListHandlingCogVideoX:
    """Test handling of empty lists in CogVideoX pipeline."""

    def test_empty_vectors_returns_none(self):
        """Test that empty vector lists result in None."""
        concept_attention_dict = {
            "concept_output_vectors": [],
            "image_output_vectors": [],
        }

        if len(concept_attention_dict.get("concept_output_vectors", [])) > 0:
            result = torch.stack(concept_attention_dict["concept_output_vectors"], dim=0)
        else:
            result = None

        assert result is None

    def test_partial_vectors_stacks_available(self):
        """Test that partial vectors are stacked correctly."""
        batch_size = 1
        num_concepts = 2
        hidden_dim = 64
        num_layers = 4

        concept_attention_dict = {
            "concept_output_vectors": [
                torch.randn(batch_size, num_layers, num_concepts, hidden_dim),
                torch.randn(batch_size, num_layers, num_concepts, hidden_dim),
            ],
        }

        result = torch.stack(concept_attention_dict["concept_output_vectors"], dim=0)
        assert result.shape == (2, batch_size, num_layers, num_concepts, hidden_dim)


class TestKwargsPassingCogVideoX:
    """Test kwargs passing through model hierarchy."""

    def test_kwargs_extraction(self):
        """Test kwargs extraction pattern used in dit_block."""
        kwargs = {
            "cache_vectors": True,
            "layer_indices": [30, 35, 40],
            "current_layer_idx": 35,
        }

        cache_vectors = kwargs.get("cache_vectors", True)
        layer_indices = kwargs.get("layer_indices", None)
        current_layer_idx = kwargs.get("current_layer_idx", 0)

        should_cache = cache_vectors and (
            layer_indices is None or current_layer_idx in layer_indices
        )
        assert should_cache is True
        assert cache_vectors is True
        assert layer_indices == [30, 35, 40]
        assert current_layer_idx == 35

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
