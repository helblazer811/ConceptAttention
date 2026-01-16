"""
Tests for output vectors functionality across all models.
Tests the stack_output_vectors function and conditional caching logic.
"""
import pytest
import torch
from concept_attention.utils import stack_output_vectors


class TestStackOutputVectors:
    """Tests for the stack_output_vectors utility function."""

    def test_basic_stacking(self):
        """Test basic stacking of vectors from multiple timesteps and layers."""
        # Create mock concept_attention_dicts structure:
        # List[List[Dict]] - [timesteps][layers]{vectors}
        batch_size = 2
        num_concepts = 5
        num_patches = 64
        hidden_dim = 256
        num_timesteps = 4
        num_layers = 3

        concept_attention_dicts = []
        for _ in range(num_timesteps):
            timestep_dicts = []
            for _ in range(num_layers):
                layer_dict = {
                    "concept_output_vectors": torch.randn(batch_size, num_concepts, hidden_dim),
                    "image_output_vectors": torch.randn(batch_size, num_patches, hidden_dim),
                }
                timestep_dicts.append(layer_dict)
            concept_attention_dicts.append(timestep_dicts)

        # Stack concept vectors
        concept_vectors = stack_output_vectors(concept_attention_dicts, "concept_output_vectors")
        assert concept_vectors is not None
        # Expected shape: (batch, time, layers, tokens, dim)
        assert concept_vectors.shape == (batch_size, num_timesteps, num_layers, num_concepts, hidden_dim)

        # Stack image vectors
        image_vectors = stack_output_vectors(concept_attention_dicts, "image_output_vectors")
        assert image_vectors is not None
        assert image_vectors.shape == (batch_size, num_timesteps, num_layers, num_patches, hidden_dim)

    def test_empty_dicts(self):
        """Test stacking with empty dicts returns None."""
        concept_attention_dicts = []
        result = stack_output_vectors(concept_attention_dicts, "concept_output_vectors")
        assert result is None

    def test_missing_key(self):
        """Test stacking when key is not present in dicts."""
        concept_attention_dicts = [
            [{"other_key": torch.randn(2, 5, 256)}]
        ]
        result = stack_output_vectors(concept_attention_dicts, "concept_output_vectors")
        assert result is None

    def test_partial_keys(self):
        """Test stacking when only some layers have the key."""
        batch_size = 2
        num_concepts = 5
        hidden_dim = 256

        concept_attention_dicts = [
            [
                {"concept_output_vectors": torch.randn(batch_size, num_concepts, hidden_dim)},
                {},  # Missing key
                {"concept_output_vectors": torch.randn(batch_size, num_concepts, hidden_dim)},
            ]
        ]

        result = stack_output_vectors(concept_attention_dicts, "concept_output_vectors")
        # Should only stack the layers that have the key
        assert result is not None
        # 1 timestep, 2 layers with the key
        assert result.shape == (batch_size, 1, 2, num_concepts, hidden_dim)


class TestConditionalCaching:
    """Tests for conditional caching logic used in all models."""

    def test_should_cache_all_layers(self):
        """Test that all layers are cached when layer_indices is None."""
        cache_vectors = True
        layer_indices = None

        for current_layer_idx in range(19):
            should_cache = cache_vectors and (
                layer_indices is None or current_layer_idx in layer_indices
            )
            assert should_cache is True

    def test_should_cache_specific_layers(self):
        """Test that only specified layers are cached."""
        cache_vectors = True
        layer_indices = [15, 16, 17, 18]

        for current_layer_idx in range(19):
            should_cache = cache_vectors and (
                layer_indices is None or current_layer_idx in layer_indices
            )
            if current_layer_idx in layer_indices:
                assert should_cache is True
            else:
                assert should_cache is False

    def test_cache_vectors_false(self):
        """Test that nothing is cached when cache_vectors is False."""
        cache_vectors = False
        layer_indices = None

        for current_layer_idx in range(19):
            should_cache = cache_vectors and (
                layer_indices is None or current_layer_idx in layer_indices
            )
            assert should_cache is False

    def test_cache_vectors_false_with_layer_indices(self):
        """Test that nothing is cached when cache_vectors is False even with layer_indices."""
        cache_vectors = False
        layer_indices = [15, 16, 17, 18]

        for current_layer_idx in range(19):
            should_cache = cache_vectors and (
                layer_indices is None or current_layer_idx in layer_indices
            )
            assert should_cache is False


class TestTimestepFiltering:
    """Tests for timestep filtering logic."""

    def test_all_timesteps_tracked(self):
        """Test that all timesteps are tracked when timestep_indices is None."""
        timestep_indices = None
        num_timesteps = 10

        for step_idx in range(num_timesteps):
            should_track = timestep_indices is None or step_idx in timestep_indices
            assert should_track is True

    def test_specific_timesteps_tracked(self):
        """Test that only specified timesteps are tracked."""
        timestep_indices = [0, 2, 4]
        num_timesteps = 10

        for step_idx in range(num_timesteps):
            should_track = timestep_indices is None or step_idx in timestep_indices
            if step_idx in timestep_indices:
                assert should_track is True
            else:
                assert should_track is False


class TestOutputVectorShapes:
    """Tests for expected output vector shapes with different configurations."""

    def test_shape_with_subset_layers(self):
        """Test output shape when only subset of layers are cached."""
        batch_size = 1
        num_concepts = 3
        num_patches = 64
        hidden_dim = 256
        num_timesteps = 4
        total_layers = 19
        cached_layers = [15, 16, 17, 18]

        concept_attention_dicts = []
        for _ in range(num_timesteps):
            timestep_dicts = []
            for layer_idx in range(total_layers):
                if layer_idx in cached_layers:
                    layer_dict = {
                        "concept_output_vectors": torch.randn(batch_size, num_concepts, hidden_dim),
                        "image_output_vectors": torch.randn(batch_size, num_patches, hidden_dim),
                    }
                else:
                    layer_dict = {}  # Not cached
                timestep_dicts.append(layer_dict)
            concept_attention_dicts.append(timestep_dicts)

        concept_vectors = stack_output_vectors(concept_attention_dicts, "concept_output_vectors")
        assert concept_vectors is not None
        # Should only have 4 layers (the cached ones)
        assert concept_vectors.shape == (batch_size, num_timesteps, len(cached_layers), num_concepts, hidden_dim)

    def test_shape_with_subset_timesteps(self):
        """Test output shape when only subset of timesteps are tracked."""
        batch_size = 1
        num_concepts = 3
        num_patches = 64
        hidden_dim = 256
        tracked_timesteps = [0, 2]  # Only track these
        num_layers = 4

        concept_attention_dicts = []
        for _ in tracked_timesteps:  # Only tracked timesteps are in the list
            timestep_dicts = []
            for _ in range(num_layers):
                layer_dict = {
                    "concept_output_vectors": torch.randn(batch_size, num_concepts, hidden_dim),
                    "image_output_vectors": torch.randn(batch_size, num_patches, hidden_dim),
                }
                timestep_dicts.append(layer_dict)
            concept_attention_dicts.append(timestep_dicts)

        concept_vectors = stack_output_vectors(concept_attention_dicts, "concept_output_vectors")
        assert concept_vectors is not None
        assert concept_vectors.shape == (batch_size, len(tracked_timesteps), num_layers, num_concepts, hidden_dim)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
