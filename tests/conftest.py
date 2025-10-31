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


# Workflow testing fixtures

@pytest.fixture
def workflow_test_image():
    """
    Path to a test image for workflow execution tests.

    Returns the path to a simple test image that can be used as input
    to image-to-3D workflows.
    """
    # Create a simple test image if it doesn't exist
    from pathlib import Path
    test_image_path = Path(__file__).parent / "test_data" / "test_cube.png"

    if not test_image_path.exists():
        # Create test_data directory if needed
        test_image_path.parent.mkdir(exist_ok=True)

        # Create a simple 512x512 test image (white cube on black background)
        img = Image.new('RGB', (512, 512), color='black')
        # Draw a white square in the center
        from PIL import ImageDraw
        draw = ImageDraw.Draw(img)
        draw.rectangle([156, 156, 356, 356], fill='white')
        img.save(test_image_path)

    return str(test_image_path)


@pytest.fixture
def load_workflow_json():
    """
    Helper fixture to load workflow JSON files.

    Usage:
        workflow = load_workflow_json("trellis-i2m.json")
    """
    import json
    from pathlib import Path

    def _load(workflow_name: str):
        workflow_path = Path(__file__).parent.parent / "workflows" / workflow_name
        with open(workflow_path) as f:
            return json.load(f)

    return _load


@pytest.fixture
def modify_workflow_attention():
    """
    Helper fixture to modify attention mechanism settings in workflows.

    Usage:
        modified = modify_workflow_attention(workflow, "xformers", "xformers-native")
    """
    import copy

    def _modify(workflow: dict, sparse_attn: str, slat_attn: str):
        """
        Modify attention settings in Load_Trellis_Model nodes.

        Args:
            workflow: Workflow dict to modify
            sparse_attn: Value for sparse_attn_impl
                        ("flash-attn", "xformers", "torch-native")
            slat_attn: Value for slat_attn_impl
                      ("flash-native", "xformers-native", etc.)

        Returns:
            Modified copy of workflow dict
        """
        workflow_copy = copy.deepcopy(workflow)

        for node in workflow_copy.get("nodes", []):
            if node.get("type") == "Load_Trellis_Model":
                # widgets_values: [model_type, sparse_attn_impl, slat_attn_impl]
                if len(node.get("widgets_values", [])) >= 3:
                    node["widgets_values"][1] = sparse_attn
                    node["widgets_values"][2] = slat_attn

        return workflow_copy

    return _modify
