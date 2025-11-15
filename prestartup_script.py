"""
ComfyUI-HunyuanX Prestartup Script

This script runs BEFORE ComfyUI loads on EVERY startup.
It performs fast runtime checks and environment setup.
Also copies example assets and workflows to ComfyUI directories on first run.

For one-time installation (dependencies, compilation, etc.), see install.py.
This separation ensures fast startup times (<2 seconds instead of 90+ seconds).
"""

import os
import sys
import site
import shutil
import glob

# Import folder_paths from ComfyUI
try:
    import folder_paths
except ImportError:
    folder_paths = None


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
                print(f"✅ ComfyUI-HunyuanX: Added PyTorch libs to LD_LIBRARY_PATH: {torch_lib_path}")


def check_cuda_extension():
    """Check if custom_rasterizer CUDA extension is compiled (handles both regular and editable installs)"""
    # Strategy 1: Try direct import (most reliable - works for all install types)
    try:
        import custom_rasterizer
        return True
    except ImportError:
        pass

    # Strategy 2: Check site-packages for regular install
    site_packages = site.getsitepackages()[0] if site.getsitepackages() else None
    if site_packages:
        # Check for regular install directory
        pkg_dir = os.path.join(site_packages, "custom_rasterizer")
        if os.path.exists(pkg_dir):
            return True

        # Check for editable install marker (.egg-link)
        egg_link = os.path.join(site_packages, "custom_rasterizer.egg-link")
        if os.path.exists(egg_link):
            try:
                # Verify the source directory has compiled .so files
                with open(egg_link, 'r') as f:
                    source_dir = f.readline().strip()
                so_files = glob.glob(os.path.join(source_dir, "**", "*.so"), recursive=True)
                if so_files:
                    return True
            except Exception:
                pass

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


def get_node_dir():
    """Get the ComfyUI-HunyuanX node directory"""
    return os.path.dirname(os.path.abspath(__file__))


def get_marker_file():
    """Get path to marker file that tracks if assets have been copied"""
    return os.path.join(get_node_dir(), ".assets_copied")


def has_assets_been_copied():
    """Check if assets have already been copied"""
    return os.path.exists(get_marker_file())


def mark_assets_as_copied():
    """Create marker file to indicate assets have been copied"""
    try:
        with open(get_marker_file(), 'w') as f:
            f.write("Assets copied successfully\n")
    except Exception as e:
        print(f"[ComfyUI-HunyuanX] Warning: Could not create marker file: {e}")


def copy_assets():
    """Copy example assets to ComfyUI input directory"""
    if not folder_paths:
        print("[ComfyUI-HunyuanX] Warning: Could not import folder_paths, skipping asset copy")
        return False

    node_dir = get_node_dir()
    assets_src = os.path.join(node_dir, "assets")

    if not os.path.exists(assets_src):
        print("[ComfyUI-HunyuanX] Warning: assets folder not found")
        return False

    # Get ComfyUI input directory with fallback
    try:
        input_dir = folder_paths.get_input_directory()
        if not input_dir:
            raise ValueError("folder_paths.get_input_directory() returned None")
    except Exception as e:
        print(f"[ComfyUI-HunyuanX] Warning: Could not get input directory: {e}")
        # Fallback: try to find ComfyUI/input relative to custom_nodes
        try:
            custom_nodes_dir = os.path.dirname(node_dir)
            comfyui_dir = os.path.dirname(custom_nodes_dir)
            input_dir = os.path.join(comfyui_dir, "input")
            if not os.path.exists(input_dir):
                print(f"[ComfyUI-HunyuanX] Warning: Fallback input directory does not exist")
                return False
        except Exception:
            return False

    try:
        copied_count = 0
        skipped_count = 0

        for filename in os.listdir(assets_src):
            # Skip checkpoint directories and hidden files
            if filename.startswith('.'):
                continue

            src_path = os.path.join(assets_src, filename)

            # Skip directories
            if os.path.isdir(src_path):
                continue

            dst_path = os.path.join(input_dir, filename)

            if os.path.exists(dst_path):
                skipped_count += 1
            else:
                shutil.copy2(src_path, dst_path)
                file_size = os.path.getsize(src_path) / 1024
                print(f"[ComfyUI-HunyuanX] ✅ Copied asset: {filename} ({file_size:.1f} KB)")
                copied_count += 1

        if copied_count > 0:
            print(f"[ComfyUI-HunyuanX] ✅ Copied {copied_count} asset(s) to {input_dir}")
        if skipped_count > 0:
            print(f"[ComfyUI-HunyuanX] ℹ️  Skipped {skipped_count} asset(s) (already exist)")

        return True

    except Exception as e:
        print(f"[ComfyUI-HunyuanX] ❌ Error copying assets: {e}")
        return False


def copy_workflows():
    """Copy example workflows to ComfyUI user workflows directory"""
    if not folder_paths:
        print("[ComfyUI-HunyuanX] Warning: Could not import folder_paths, skipping workflow copy")
        return False

    node_dir = get_node_dir()
    workflows_src = os.path.join(node_dir, "workflows")

    if not os.path.exists(workflows_src):
        print("[ComfyUI-HunyuanX] Warning: workflows folder not found")
        return False

    # Get ComfyUI user workflows directory
    try:
        user_dir = folder_paths.get_user_directory()
        workflows_dst = os.path.join(user_dir, "default", "workflows")
        os.makedirs(workflows_dst, exist_ok=True)
    except Exception as e:
        print(f"[ComfyUI-HunyuanX] Warning: Could not get workflows directory: {e}")
        return False

    try:
        copied_count = 0

        for filename in os.listdir(workflows_src):
            if not filename.endswith('.json'):
                continue

            src_path = os.path.join(workflows_src, filename)
            # Prefix workflow names to avoid conflicts
            dst_filename = f"HunyuanX_{filename}"
            dst_path = os.path.join(workflows_dst, dst_filename)

            # Skip if file already exists (don't overwrite user workflows)
            if os.path.exists(dst_path):
                continue

            shutil.copy2(src_path, dst_path)
            copied_count += 1
            print(f"[ComfyUI-HunyuanX] ✅ Copied workflow: {dst_filename}")

        if copied_count > 0:
            print(f"[ComfyUI-HunyuanX] ✅ Copied {copied_count} workflow(s) to {workflows_dst}")

        return True

    except Exception as e:
        print(f"[ComfyUI-HunyuanX] ❌ Error copying workflows: {e}")
        return False


def print_startup_status():
    """Print startup status and any warnings about missing components"""
    print("\n" + "="*80)
    print("🔧 ComfyUI-HunyuanX: Checking environment...")
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

    # Print helpful message if anything is missing
    if has_warnings:
        print("\n" + "-"*80)
        print("⚠️  Some components are missing!")
        print("   To install all dependencies and compile extensions, run:")
        print("   cd /path/to/ComfyUI/custom_nodes/ComfyUI-HunyuanX")
        print("   python install.py")
        print("\n   Or use ComfyUI-Manager to reinstall this custom node.")
        print("-"*80)
    else:
        print("\n✅ All Hunyuan 3D components ready!")

    print("\n" + "="*80 + "\n")


# Run on import (only when loaded as module, not when executed directly)
if __name__ != "__main__":
    # Set up library paths first (required for CUDA extensions to load)
    setup_torch_library_path()

    # Copy assets and workflows on first run (skips files that already exist)
    if not has_assets_been_copied():
        print("[ComfyUI-HunyuanX] First run detected, copying assets and workflows...")
        assets_ok = copy_assets()
        workflows_ok = copy_workflows()

        if assets_ok or workflows_ok:
            mark_assets_as_copied()
            print("[ComfyUI-HunyuanX] ✅ Asset and workflow setup complete\n")

    # Print status and check for missing components
    print_startup_status()
