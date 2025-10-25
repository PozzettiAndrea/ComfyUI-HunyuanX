"""
Test that fixtures are working correctly.

This is a sanity check to ensure the test infrastructure is set up properly.
"""

import pytest
import torch
import numpy as np


class TestFixtures:
    """Test that all fixtures load correctly."""

    def test_sample_trimesh_fixture(self, sample_trimesh):
        """Test that sample_trimesh fixture creates a valid mesh."""
        assert sample_trimesh is not None
        assert hasattr(sample_trimesh, 'vertices')
        assert hasattr(sample_trimesh, 'faces')
        assert len(sample_trimesh.vertices) > 0
        assert len(sample_trimesh.faces) > 0

    def test_sample_image_tensor_fixture(self, sample_image_tensor):
        """Test that sample_image_tensor fixture creates valid tensor."""
        assert isinstance(sample_image_tensor, torch.Tensor)
        assert sample_image_tensor.shape == (1, 512, 512, 3)
        assert sample_image_tensor.min() >= 0.0
        assert sample_image_tensor.max() <= 1.0

    def test_sample_pil_image_fixture(self, sample_pil_image):
        """Test that sample_pil_image fixture creates valid PIL image."""
        from PIL import Image
        assert isinstance(sample_pil_image, Image.Image)
        assert sample_pil_image.size == (512, 512)
        assert sample_pil_image.mode == 'RGB'

    def test_temp_mesh_file_fixture(self, temp_mesh_file):
        """Test that temp_mesh_file fixture creates a valid file."""
        import os
        assert os.path.exists(temp_mesh_file)
        assert temp_mesh_file.endswith('.obj')

    def test_cpu_only_fixture(self):
        """Test that CPU-only fixture forces CPU usage."""
        # This fixture is autouse, so it should always be active
        assert not torch.cuda.is_available()


class TestHelperFunctions:
    """Test helper functions from conftest."""

    def test_create_test_mesh(self):
        """Test create_test_mesh helper."""
        from tests.conftest import create_test_mesh

        mesh = create_test_mesh()
        assert mesh is not None
        assert len(mesh.vertices) == 8  # Cube has 8 vertices
        assert len(mesh.faces) == 12  # Cube has 12 triangular faces

    def test_tensor_to_pil(self, sample_image_tensor):
        """Test tensor_to_pil helper."""
        from tests.conftest import tensor_to_pil

        pil_img = tensor_to_pil(sample_image_tensor)
        assert pil_img.size == (512, 512)
        assert pil_img.mode == 'RGB'

    def test_pil_to_tensor(self, sample_pil_image):
        """Test pil_to_tensor helper."""
        from tests.conftest import pil_to_tensor

        tensor = pil_to_tensor(sample_pil_image)
        assert tensor.shape == (1, 512, 512, 3)
        assert tensor.min() >= 0.0
        assert tensor.max() <= 1.0

    def test_round_trip_conversion(self, sample_image_tensor):
        """Test that tensor->PIL->tensor round trip preserves shape."""
        from tests.conftest import tensor_to_pil, pil_to_tensor

        pil_img = tensor_to_pil(sample_image_tensor)
        tensor_again = pil_to_tensor(pil_img)

        assert tensor_again.shape == sample_image_tensor.shape
