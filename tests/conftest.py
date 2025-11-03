import sys
import types
import pytest
import os
import shutil
from pathlib import Path

# --- Add MeshCraft to sys.path ---
TESTS_DIR = Path(__file__).resolve().parent
MESHCRAFT_DIR = TESTS_DIR.parent
if str(MESHCRAFT_DIR) not in sys.path:
    sys.path.insert(0, str(MESHCRAFT_DIR))
print(f"✅ Added MeshCraft to sys.path: {MESHCRAFT_DIR}")

# --- Force override any real comfy modules ---
for m in ["comfy", "comfy.utils", "comfy.model_management"]:
    if m in sys.modules:
        del sys.modules[m]

# --- Create ComfyUI mocks ---
mock_comfy_utils = types.SimpleNamespace(
    common_upscale=lambda *a, **kw: None,
    ProgressBar=lambda total=0: types.SimpleNamespace(
        update=lambda *a, **kw: None, close=lambda *a, **kw: None
    ),
    load_torch_file=lambda path: None,
)

mock_model_management = types.SimpleNamespace(
    get_torch_device=lambda: "cpu",
    unet_offload_device=lambda: "cpu",
    soft_empty_cache=lambda: None,
)

mock_folder_paths = types.SimpleNamespace(
    get_input_directory=lambda: "input",
    get_output_directory=lambda: "output",
    get_full_path=lambda *a, **kw: "dummy/path",
    get_filename_list=lambda *a, **kw: ["dummy_model.safetensors"],
    get_save_image_path=lambda *a, **kw: ("output", "file", 0, "output", "file"),
    filter_files_content_types=lambda files, types: files,
    get_annotated_filepath=lambda f: f,
    exists_annotated_filepath=lambda f: True,
)

mock_node_helpers = types.SimpleNamespace(
    pillow=lambda f, *a, **kw: f(*a, **kw)
)

# --- Register mocks globally ---
sys.modules.update({
    "comfy": types.SimpleNamespace(utils=mock_comfy_utils, model_management=mock_model_management),
    "comfy.utils": mock_comfy_utils,
    "comfy.model_management": mock_model_management,
    "folder_paths": mock_folder_paths,
    "node_helpers": mock_node_helpers,
})

print("✅ Mocked comfy.*, folder_paths, node_helpers for pytest runtime")

# --- Optional: ensure pytest fixture context stays intact ---
@pytest.fixture(scope="session", autouse=True)
def comfy_mocks_loaded():
    """Ensures comfy mocks are loaded before any test."""
    return True

# --- Setup test images ---
@pytest.fixture(scope="session", autouse=True)
def setup_test_images():
    """Copy test images from examples/ to ComfyUI input/ folder before tests run"""
    # Get paths
    meshcraft_root = MESHCRAFT_DIR
    examples_dir = meshcraft_root / "examples"

    # ComfyUI input directory (3 levels up from MeshCraft: custom_nodes -> ComfyUI, then into input/)
    comfyui_root = meshcraft_root.parent.parent
    input_dir = comfyui_root / "input"

    # Ensure input directory exists
    input_dir.mkdir(parents=True, exist_ok=True)

    # List of test images to copy
    test_images = [
        "typical_building_hunyuan.png",
        "typical_building_trellis.png",
    ]

    print(f"\n📋 Copying test images from {examples_dir} to {input_dir}")

    copied_files = []
    for image_name in test_images:
        src = examples_dir / image_name
        dst = input_dir / image_name

        if src.exists():
            shutil.copy2(src, dst)
            copied_files.append(dst)
            print(f"  ✅ Copied: {image_name}")
        else:
            print(f"  ⚠️  Not found: {image_name}")

    yield

    # Optional: Clean up after tests complete
    # Uncomment if you want to remove test images after tests finish
    # for dst in copied_files:
    #     if dst.exists():
    #         dst.unlink()
    #         print(f"  🗑️  Cleaned up: {dst.name}")

def pytest_addoption(parser):
    parser.addoption(
        "--config",
        action="store",
        default=None,
        help="Path to workflow config file (optional, used for workflow tests)",
    )

import trimesh
import tempfile
import numpy as np

@pytest.fixture
def temp_mesh_file(tmp_path):
    """Create a temporary .obj mesh file for testing."""
    mesh = trimesh.creation.box(extents=(1, 1, 1))
    mesh_path = tmp_path / "test_mesh.obj"
    mesh.export(mesh_path)
    return mesh_path

@pytest.fixture
def sample_trimesh():
    """Provide a simple cube mesh for postprocessing tests."""
    return trimesh.creation.box(extents=(1, 1, 1))

@pytest.fixture(scope="class")
def args_pytest():
    """
    Dummy args fixture for workflow tests.
    Used to provide --listen and --port arguments for ComfyUI server connection.
    """
    return {"listen": "0.0.0.0", "port": 8188}

@pytest.fixture
def performance_tracker():
    """Mocked performance tracker to avoid dependency on testutils.metrics."""
    class DummyTracker:
        def add_outputs(self, outputs):
            print(f"📦 Tracking {len(outputs)} outputs")

        def print_summary(self):
            print("📊 Mock summary — no real performance data recorded.")

        def save_results(self, format="both"):
            print("💾 Mock saving results (no files created)")
            return {"json": "/tmp/mock_results.json", "csv": "/tmp/mock_results.csv"}

    return DummyTracker()


@pytest.fixture
def track_workflow_performance():
    """Context manager mock for performance tracking."""
    from contextlib import contextmanager

    @contextmanager
    def _tracker(**kwargs):
        class Dummy:
            def add_outputs(self, outputs): pass
        print(f"⚙️  Mock tracking workflow performance: {kwargs}")
        yield Dummy()
    return _tracker