"""
Pytest configuration and shared fixtures for ComfyUI-MeshCraft tests.

This file provides common test fixtures and setup for all tests.
"""

import pytest
import torch
import numpy as np
from unittest.mock import MagicMock, patch
from PIL import Image


@pytest.fixture
def mock_comfy_modules():
    """
    Mock ComfyUI modules to prevent CUDA initialization and import errors during testing.

    Similar to how ComfyUI's own tests mock their modules.
    """
    mock_nodes = MagicMock()
    mock_nodes.MAX_RESOLUTION = 16384

    mock_server = MagicMock()
    mock_folder_paths = MagicMock()

    with patch.dict('sys.modules', {
        'nodes': mock_nodes,
        'server': mock_server,
        'folder_paths': mock_folder_paths,
    }):
        yield


@pytest.fixture
def sample_trimesh():
    """Create a simple test mesh (cube) for testing."""
    try:
        import trimesh
        # Create a simple cube mesh
        mesh = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
        return mesh
    except ImportError:
        pytest.skip("trimesh not installed")


@pytest.fixture
def sample_image_tensor():
    """Create a sample image tensor in ComfyUI format (B, H, W, C)."""
    # ComfyUI uses BHWC format with values in range [0, 1]
    return torch.rand(1, 512, 512, 3)


@pytest.fixture
def sample_pil_image():
    """Create a sample PIL image."""
    # Create a 512x512 RGB image with random data
    img_array = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    return Image.fromarray(img_array)


@pytest.fixture
def sample_mesh_with_uvs():
    """Create a test mesh with UV coordinates."""
    try:
        import trimesh
        # Create a box with UV coordinates
        mesh = trimesh.creation.box(extents=[1.0, 1.0, 1.0])

        # Add simple UV coordinates (normally would be unwrapped properly)
        # For testing, just create placeholder UVs
        num_vertices = len(mesh.vertices)
        mesh.visual.uv = np.random.rand(num_vertices, 2)

        return mesh
    except ImportError:
        pytest.skip("trimesh not installed")


@pytest.fixture
def temp_mesh_file(tmp_path):
    """Create a temporary mesh file for I/O testing."""
    import trimesh

    # Create a simple mesh
    mesh = trimesh.creation.box(extents=[1.0, 1.0, 1.0])

    # Save to temporary file
    mesh_path = tmp_path / "test_mesh.obj"
    mesh.export(str(mesh_path))

    return str(mesh_path)


@pytest.fixture(autouse=True)
def use_cpu_only():
    """
    Force all tests to use CPU, even if CUDA is available.

    This prevents GPU memory issues and makes tests faster/more reproducible.
    """
    original_device = torch.cuda.is_available
    torch.cuda.is_available = lambda: False
    yield
    torch.cuda.is_available = original_device


# Helper functions for tests

def create_test_mesh(vertices_count=8, faces_count=12):
    """Helper to create a simple test mesh with specified complexity."""
    import trimesh

    if vertices_count == 8 and faces_count == 12:
        # Default cube
        return trimesh.creation.box(extents=[1.0, 1.0, 1.0])
    else:
        # Create a UV sphere with approximate vertex count
        subdivisions = max(0, int(np.log2(vertices_count / 8)))
        return trimesh.creation.icosphere(subdivisions=subdivisions)


def tensor_to_pil(tensor):
    """Convert ComfyUI tensor format (BHWC) to PIL Image."""
    # Remove batch dimension and convert to numpy
    img_np = tensor[0].numpy()
    # Convert from [0, 1] to [0, 255]
    img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(img_np)


def pil_to_tensor(pil_image):
    """Convert PIL Image to ComfyUI tensor format (BHWC)."""
    img_np = np.array(pil_image).astype(np.float32) / 255.0
    # Add batch dimension
    return torch.from_numpy(img_np).unsqueeze(0)
