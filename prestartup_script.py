"""
ComfyUI-MeshCraft Prestartup Script

This script runs BEFORE ComfyUI loads on EVERY startup.
It performs fast runtime checks and environment setup.

For one-time installation (dependencies, compilation, etc.), see scripts/install.py.
This separation ensures fast startup times (<2 seconds instead of 90+ seconds).
"""

import os
import sys
import site
import shutil
import glob


def setup_torch_library_path():
    """
    Add PyTorch library path to LD_LIBRARY_PATH so CUDA extensions can find libc10.so, etc.

    This is needed because CUDA extensions link against PyTorch's C++ libraries.
    We cannot import torch here as it must not be imported before ComfyUI initializes.
    """
    # Find torch library path without importing torch
    site_packages = site.getsitepackages()[0] if site.getsitepackages() else None

    if not site_packages:
        # Fallback: try to find site-packages from Python executable
        python_dir = os.path.dirname(sys.executable)
        possible_paths = [
            os.path.join(python_dir, '..', 'lib', f'python{sys.version_info.major}.{sys.version_info.minor}', 'site-packages'),
            os.path.join(python_dir, '..', 'lib', 'site-packages'),
        ]
        for path in possible_paths:
            if os.path.exists(path):
                site_packages = os.path.abspath(path)
                break

    if site_packages:
        torch_lib_path = os.path.join(site_packages, 'torch', 'lib')

        if os.path.exists(torch_lib_path):
            current_ld_path = os.environ.get('LD_LIBRARY_PATH', '')
            if torch_lib_path not in current_ld_path:
                os.environ['LD_LIBRARY_PATH'] = f"{torch_lib_path}:{current_ld_path}"
                print(f"✅ ComfyUI-MeshCraft: Added PyTorch libs to LD_LIBRARY_PATH: {torch_lib_path}")


def check_cuda_extension():
    """Check if custom_rasterizer CUDA extension is compiled (fast check only)"""
    site_packages = site.getsitepackages()[0] if site.getsitepackages() else None
    if site_packages:
        pkg_dir = os.path.join(site_packages, "custom_rasterizer")
        if os.path.exists(pkg_dir):
            return True
    return False


def check_mesh_inpaint_processor():
    """Check if mesh_inpaint_processor is compiled (fast check only)"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    renderer_dir = os.path.join(current_dir, "nodes", "lib", "hy3dpaint", "DifferentiableRenderer")
    so_files = glob.glob(os.path.join(renderer_dir, "mesh_inpaint_processor*.so"))
    return len(so_files) > 0


def check_blender():
    """Check if Blender is available (fast check only)"""
    blender_path = os.environ.get("BLENDER_PATH")
    if blender_path and os.path.exists(blender_path):
        return True

    if shutil.which("blender"):
        return True

    common_paths = [
        "/usr/bin/blender",
        "/usr/local/bin/blender",
        "/Applications/Blender.app/Contents/MacOS/Blender",
    ]
    for path in common_paths:
        if os.path.exists(path):
            return True

    return False


def check_trellis_package():
    """Check if TRELLIS package exists (fast directory check only)"""
    trellis_path = os.path.join(os.path.dirname(__file__), "nodes", "lib", "trellis")
    trellis_pipeline_file = os.path.join(trellis_path, "pipelines", "trellis_image_to_3d.py")
    return os.path.exists(trellis_pipeline_file)


def check_trellis_dependencies():
    """Check if TRELLIS Python dependencies are installed (fast check only)"""
    dependencies = ["imageio", "imageio_ffmpeg", "einops"]

    for package in dependencies:
        try:
            __import__(package)
        except ImportError:
            return False

    return True


def print_startup_status():
    """Print startup status and any warnings about missing components"""
    print("\n" + "="*80)
    print("🔧 ComfyUI-MeshCraft: Checking environment...")
    print("="*80 + "\n")

    # Track if anything is missing
    has_warnings = False

    # Check CUDA extension
    if check_cuda_extension():
        print("✅ custom_rasterizer CUDA extension")
    else:
        print("⚠️  custom_rasterizer not compiled (texture generation may be limited)")
        has_warnings = True

    # Check mesh inpaint processor
    if check_mesh_inpaint_processor():
        so_files = glob.glob(os.path.join(
            os.path.dirname(__file__),
            "nodes", "lib", "hy3dpaint", "DifferentiableRenderer",
            "mesh_inpaint_processor*.so"
        ))
        print(f"✅ mesh_inpaint_processor: {os.path.basename(so_files[0])}")
    else:
        print("⚠️  mesh_inpaint_processor not compiled (texture inpainting may not work)")
        has_warnings = True

    # Check Blender
    if check_blender():
        print("✅ Blender found in PATH")
    else:
        print("⚠️  Blender not found (RGB multiview rendering unavailable)")
        has_warnings = True

    # Check TRELLIS package
    if check_trellis_package():
        print("✅ TRELLIS package found (local repository)")
    else:
        print("⚠️  TRELLIS package not found")
        has_warnings = True

    # Check TRELLIS dependencies
    if check_trellis_dependencies():
        print("✅ TRELLIS dependencies installed")
    else:
        print("⚠️  Some TRELLIS dependencies missing")
        has_warnings = True

    # Print helpful message if anything is missing
    if has_warnings:
        print("\n" + "-"*80)
        print("⚠️  Some components are missing!")
        print("   To install all dependencies and compile extensions, run:")
        print("   cd /workspace/.../ComfyUI-MeshCraft")
        print("   python scripts/install.py")
        print("\n   Or use ComfyUI-Manager to reinstall this custom node.")
        print("-"*80)

    print("\n" + "="*80)
    print("✅ All basic TRELLIS dependencies installed!")
    print("="*80 + "\n")


# Run on import (only when loaded as module, not when executed directly)
if __name__ != "__main__":
    # Set up library paths first (required for CUDA extensions to load)
    setup_torch_library_path()

    # Print status and check for missing components
    print_startup_status()
