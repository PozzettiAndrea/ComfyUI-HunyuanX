"""
Basic tests for ComfyUI-MeshCraft nodes.

These tests demonstrate testing patterns for custom ComfyUI nodes:
- Input validation
- Basic functionality
- Edge cases
- Mocking heavy operations
"""

import pytest
import torch
import numpy as np
from unittest.mock import MagicMock, patch


class TestMeshLoadAndExport:
    """Tests for mesh loading and exporting nodes."""

    def test_load_mesh_creates_trimesh_object(self, temp_mesh_file):
        """Test that LoadMesh node correctly loads a mesh file."""
        from nodes import Hy3D21LoadMesh

        node = Hy3D21LoadMesh()
        result = node.load(temp_mesh_file)

        # Should return tuple with (trimesh, preview_image)
        assert len(result) == 2
        trimesh_obj, preview = result

        # Verify trimesh object has vertices and faces
        assert hasattr(trimesh_obj, 'vertices')
        assert hasattr(trimesh_obj, 'faces')
        assert len(trimesh_obj.vertices) > 0
        assert len(trimesh_obj.faces) > 0

    def test_load_nonexistent_mesh_raises_error(self):
        """Test that loading a non-existent file raises an error."""
        from nodes import Hy3D21LoadMesh

        node = Hy3D21LoadMesh()

        with pytest.raises(Exception):
            node.load("/nonexistent/path/to/mesh.obj")


class TestMeshPostprocessing:
    """Tests for mesh post-processing operations."""

    # Decimation tests removed - decimation nodes were removed from codebase

    def test_uv_unwrap_adds_uv_coordinates(self, sample_trimesh):
        """Test that UV unwrapping adds UV coordinates to mesh."""
        from nodes import Hy3D21MeshUVWrap

        node = Hy3D21MeshUVWrap()

        # Original mesh may not have UVs
        result = node.process(sample_trimesh)

        unwrapped_mesh = result[0]

        # After unwrapping, should have UV coordinates
        assert hasattr(unwrapped_mesh.visual, 'uv')
        assert unwrapped_mesh.visual.uv is not None
        assert len(unwrapped_mesh.visual.uv) > 0

        # UV coordinates should be in [0, 1] range
        uvs = unwrapped_mesh.visual.uv
        assert np.all(uvs >= 0.0)
        assert np.all(uvs <= 1.0)



class TestCameraConfiguration:
    """Tests for camera configuration nodes."""

    def test_camera_config_parses_string_inputs(self):
        """Test that camera config correctly parses comma-separated strings."""
        from hunyuan_nodes import Hy3D21CameraConfig

        node = Hy3D21CameraConfig()

        result = node.process(
            camera_azimuths="0, 90, 180, 270",
            camera_elevations="0, 0, 0, 0",
            view_weights="1.0, 0.5, 0.5, 0.5",
            ortho_scale=1.0
        )

        camera_config = result[0]

        # Verify parsed correctly
        assert camera_config["selected_camera_azims"] == [0, 90, 180, 270]
        assert camera_config["selected_camera_elevs"] == [0, 0, 0, 0]
        assert camera_config["selected_view_weights"] == [1.0, 0.5, 0.5, 0.5]
        assert camera_config["ortho_scale"] == 1.0

    def test_camera_config_handles_spaces_in_input(self):
        """Test that camera config handles various spacing in input strings."""
        from hunyuan_nodes import Hy3D21CameraConfig

        node = Hy3D21CameraConfig()

        # Try with different spacing
        result = node.process(
            camera_azimuths="0,90,180",  # No spaces
            camera_elevations="0 , 0 , 0",  # Extra spaces
            view_weights="1.0,  0.5,0.5",  # Inconsistent spacing
            ortho_scale=1.5
        )

        camera_config = result[0]

        # Should still parse correctly
        assert len(camera_config["selected_camera_azims"]) == 3
        assert len(camera_config["selected_camera_elevs"]) == 3
        assert len(camera_config["selected_view_weights"]) == 3


class TestInputValidation:
    """Tests for input validation and error handling."""

    # Negative target faces test removed - decimation nodes were removed from codebase


# Mark tests that require heavy imports as slow
@pytest.mark.slow
class TestHeavyOperations:
    """
    Tests for operations that require heavy model loading.

    These are marked as 'slow' and can be skipped with: pytest -m "not slow"
    """

    @pytest.mark.skip(reason="Requires model files - for manual testing only")
    def test_texture_generation_with_mock(self):
        """
        Example of how to test texture generation with mocked model.

        This test is skipped by default as it would require model files.
        """
        from rendering_nodes import GenerateMultiviewPBR

        # Mock the heavy model loading
        with patch('rendering_nodes.LoadHunyuanMultiViewModel') as mock_loader:
            mock_model = MagicMock()
            mock_loader.return_value = mock_model

            node = GenerateMultiviewPBR()

            # Test would go here
            pass
