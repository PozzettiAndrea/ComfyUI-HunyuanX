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

    def test_mesh_decimation_reduces_face_count(self, sample_trimesh):
        """Test that mesh decimation reduces the number of faces."""
        from nodes import Hy3D21MeshlibDecimate

        node = Hy3D21MeshlibDecimate()
        original_face_count = len(sample_trimesh.faces)

        # Decimate to 50% of faces
        target_faces = original_face_count // 2
        result = node.process(sample_trimesh, target_faces)

        decimated_mesh = result[0]
        new_face_count = len(decimated_mesh.faces)

        # Should have reduced faces (might not be exact due to algorithm)
        assert new_face_count < original_face_count
        assert new_face_count <= target_faces * 1.1  # Allow 10% margin

    def test_mesh_decimation_preserves_topology(self, sample_trimesh):
        """Test that decimation preserves mesh topology (still watertight if started watertight)."""
        from nodes import Hy3D21MeshlibDecimate

        node = Hy3D21MeshlibDecimate()
        original_is_watertight = sample_trimesh.is_watertight

        target_faces = len(sample_trimesh.faces) // 2
        result = node.process(sample_trimesh, target_faces)

        decimated_mesh = result[0]

        # If original was watertight, decimated should be too (box mesh is watertight)
        if original_is_watertight:
            assert decimated_mesh.is_watertight

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


class TestImageProcessing:
    """Tests for image processing nodes."""

    def test_resize_images_changes_dimensions(self, sample_image_tensor):
        """Test that image resizing changes dimensions correctly."""
        from hunyuan_nodes import Hy3D21ResizeImages

        node = Hy3D21ResizeImages()

        # Original is 512x512
        assert sample_image_tensor.shape == (1, 512, 512, 3)

        # Resize to 256x256
        result = node.process(sample_image_tensor, 256, 256)
        resized = result[0]

        assert resized.shape == (1, 256, 256, 3)

    def test_resize_preserves_batch_and_channels(self, sample_image_tensor):
        """Test that resizing preserves batch size and channel count."""
        from hunyuan_nodes import Hy3D21ResizeImages

        node = Hy3D21ResizeImages()

        # Create a batch of 3 images
        batch_tensor = torch.cat([sample_image_tensor] * 3, dim=0)
        assert batch_tensor.shape == (3, 512, 512, 3)

        result = node.process(batch_tensor, 256, 256)
        resized = result[0]

        # Should preserve batch size and channels
        assert resized.shape[0] == 3  # Batch size
        assert resized.shape[3] == 3  # Channels
        assert resized.shape[1:3] == (256, 256)  # New dimensions


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

    def test_negative_target_faces_rejected(self, sample_trimesh):
        """Test that negative face count is handled appropriately."""
        from nodes import Hy3D21MeshlibDecimate

        node = Hy3D21MeshlibDecimate()

        # Should either raise an error or clamp to valid value
        with pytest.raises((ValueError, AssertionError)):
            node.process(sample_trimesh, -100)

    def test_zero_resolution_rejected(self, sample_image_tensor):
        """Test that zero or negative resolution is rejected."""
        from hunyuan_nodes import Hy3D21ResizeImages

        node = Hy3D21ResizeImages()

        with pytest.raises((ValueError, AssertionError)):
            node.process(sample_image_tensor, 0, 512)

        with pytest.raises((ValueError, AssertionError)):
            node.process(sample_image_tensor, 512, -1)


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
