"""
ComfyUI-MeshCraft Prestartup Script

This script runs BEFORE ComfyUI loads to compile the custom CUDA rasterizer
extension needed for Hunyuan3D texture generation.

The CUDA extension only needs to be compiled once. After that, it will be
available for import.
"""

import os
import sys
import subprocess
import site

def check_cuda_libraries():
    """
    Check if required CUDA development libraries are installed.

    Returns:
        tuple: (bool, list) - (all_available, missing_headers)
    """
    # Check for key header files that custom_rasterizer needs
    cuda_home = os.environ.get('CUDA_HOME', '/usr/local/cuda-12.8')
    required_headers = [
        'cusparse.h',      # CUDA sparse matrix library
        'cublas_v2.h',     # CUDA basic linear algebra
        'cufft.h',         # CUDA FFT library
    ]

    include_dir = os.path.join(cuda_home, 'include')
    if not os.path.exists(include_dir):
        return False, ['CUDA not found at ' + cuda_home]

    missing = []
    for header in required_headers:
        header_path = os.path.join(include_dir, header)
        if not os.path.exists(header_path):
            missing.append(header)

    return len(missing) == 0, missing


def install_cuda_libraries():
    """
    Automatically install missing CUDA development libraries.

    Returns:
        bool: True if installation succeeded, False otherwise
    """
    print("\n" + "="*80)
    print("📦 ComfyUI-MeshCraft: Installing CUDA development libraries...")
    print("   Package: cuda-libraries-dev-12-8 (~100MB)")
    print("   This is needed for custom rasterizer compilation")
    print("="*80 + "\n")

    import shutil

    # Check if running as root or if sudo is available
    try:
        is_root = os.geteuid() == 0
    except AttributeError:
        # Windows doesn't have geteuid
        print("⚠️  Auto-installation only supported on Linux")
        return False

    if not is_root and not shutil.which("sudo"):
        print("⚠️  sudo not available, cannot auto-install")
        print("   Please install manually: sudo apt-get install cuda-libraries-dev-12-8")
        return False

    cmd_prefix = [] if is_root else ["sudo", "-n"]  # -n = non-interactive

    try:
        # Update package list first
        print("   Updating package list...")
        update_result = subprocess.run(
            cmd_prefix + ["apt-get", "update", "-qq"],
            stdout=sys.stdout,
            stderr=sys.stderr,
            timeout=120
        )

        # Install cuda-libraries-dev-12-8
        print("   Installing CUDA libraries (this may take 1-2 minutes)...")
        result = subprocess.run(
            cmd_prefix + ["apt-get", "install", "-y", "cuda-libraries-dev-12-8"],
            stdout=sys.stdout,
            stderr=sys.stderr,
            timeout=300
        )

        if result.returncode == 0:
            print("\n" + "="*80)
            print("✅ ComfyUI-MeshCraft: CUDA libraries installed successfully!")
            print("="*80 + "\n")
            return True
        else:
            print("\n" + "="*80)
            print("⚠️  ComfyUI-MeshCraft: Installation failed")
            print("   You can install manually:")
            print("   sudo apt-get install cuda-libraries-dev-12-8")
            print("="*80 + "\n")
            return False

    except subprocess.TimeoutExpired:
        print("⚠️  Installation timed out")
        return False
    except Exception as e:
        print(f"⚠️  Installation error: {e}")
        return False


def compile_cuda_extension():
    """Compile the custom rasterizer CUDA extension if not already compiled."""

    # Check if already compiled by looking for the built extension in site-packages
    # We cannot import it here because that would import torch before ComfyUI initializes
    site_packages = site.getsitepackages()[0] if site.getsitepackages() else None
    if site_packages:
        # Check for common extension file patterns
        possible_extensions = [
            os.path.join(site_packages, "custom_rasterizer.so"),
            os.path.join(site_packages, "custom_rasterizer.pyd"),
            os.path.join(site_packages, "custom-rasterizer.egg-link"),
        ]
        # Also check in the custom_rasterizer package directory
        pkg_dir = os.path.join(site_packages, "custom_rasterizer")
        if os.path.exists(pkg_dir):
            print("✅ ComfyUI-MeshCraft: custom_rasterizer already compiled")
            return True

        # Check if any of the extension files exist
        for ext_path in possible_extensions:
            if os.path.exists(ext_path):
                print("✅ ComfyUI-MeshCraft: custom_rasterizer already compiled")
                return True

    # Check if CUDA libraries are available before attempting compilation
    libs_available, missing = check_cuda_libraries()
    if not libs_available:
        print("\n" + "="*80)
        print("⚠️  ComfyUI-MeshCraft: Missing CUDA development libraries")
        print(f"   Missing: {', '.join(missing)}")
        print("="*80 + "\n")

        # Try to install automatically
        if install_cuda_libraries():
            print("✅ CUDA libraries installed, proceeding with compilation...")
        else:
            print("\n" + "="*80)
            print("⚠️  ComfyUI-MeshCraft: Skipping custom rasterizer compilation")
            print("   Basic 3D mesh generation will still work")
            print("   For advanced texture features, install manually:")
            print("   sudo apt-get install cuda-libraries-dev-12-8")
            print("="*80 + "\n")
            return False

    print("\n" + "="*80)
    print("🔧 ComfyUI-MeshCraft: Compiling CUDA extension...")
    print("   This only needs to happen once and may take 1-2 minutes.")
    print("="*80 + "\n")

    # Get path to custom_rasterizer setup.py
    current_dir = os.path.dirname(os.path.abspath(__file__))
    rasterizer_dir = os.path.join(current_dir, "nodes", "lib", "hy3dpaint", "custom_rasterizer")
    setup_py = os.path.join(rasterizer_dir, "setup.py")

    if not os.path.exists(setup_py):
        print(f"⚠️  ComfyUI-MeshCraft: custom_rasterizer setup.py not found at {setup_py}")
        print("   Texture generation nodes will not work.")
        return False

    # Compile the extension
    try:
        # Change to the rasterizer directory
        original_dir = os.getcwd()
        os.chdir(rasterizer_dir)

        # Run setup.py install
        result = subprocess.run(
            [sys.executable, "setup.py", "install"],
            stdout=sys.stdout,
            stderr=sys.stderr,
            timeout=300  # 5 minute timeout
        )

        os.chdir(original_dir)

        if result.returncode == 0:
            print("\n" + "="*80)
            print("✅ ComfyUI-MeshCraft: CUDA extension compiled successfully!")
            print("="*80 + "\n")
            return True
        else:
            print("\n" + "="*80)
            print("❌ ComfyUI-MeshCraft: CUDA extension compilation failed!")
            print("   Check the output above for details")
            print("="*80 + "\n")
            return False

    except subprocess.TimeoutExpired:
        print("❌ ComfyUI-MeshCraft: Compilation timed out after 5 minutes")
        os.chdir(original_dir)
        return False
    except Exception as e:
        print(f"❌ ComfyUI-MeshCraft: Compilation error: {e}")
        os.chdir(original_dir)
        return False


def setup_torch_library_path():
    """
    Add PyTorch library path to LD_LIBRARY_PATH so CUDA extensions can find libc10.so, etc.

    This is needed because CUDA extensions link against PyTorch's C++ libraries.
    We cannot import torch here as it must not be imported before ComfyUI initializes.
    """
    # Find torch library path without importing torch
    # Construct path from site-packages
    site_packages = site.getsitepackages()[0] if site.getsitepackages() else None
    if not site_packages:
        # Fallback: try to find site-packages from Python executable
        python_dir = os.path.dirname(sys.executable)
        # Common patterns: .../bin/python -> .../lib/pythonX.Y/site-packages
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


def install_flash_attention():
    """
    Install Flash Attention 2 for optimal diffusion model performance.

    Flash Attention provides 10-20% faster inference on modern GPUs (RTX 4090/5090, etc.)
    Falls back gracefully to SDPA if installation fails.
    """
    print("\n" + "="*80)
    print("🚀 ComfyUI-MeshCraft: Checking Flash Attention...")
    print("="*80 + "\n")

    # Check if already installed
    try:
        import flash_attn
        print(f"✅ Flash Attention {flash_attn.__version__} already installed")
        return True
    except ImportError:
        pass

    print("   Flash Attention not found, attempting installation...")
    print("   This provides 10-20% faster inference on modern GPUs")
    print("   (Compilation may take 3-5 minutes)\n")

    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "flash-attn==2.8.2", "--no-build-isolation"],
            stdout=sys.stdout,
            stderr=sys.stderr,
            timeout=600  # 10 minute timeout for compilation
        )

        if result.returncode == 0:
            print("\n" + "="*80)
            print("✅ Flash Attention installed successfully!")
            print("   Your multiview model will use Flash Attention for faster inference")
            print("="*80 + "\n")
            return True
        else:
            print("\n" + "="*80)
            print("⚠️  Flash Attention installation failed (non-critical)")
            print("   Model will fall back to SDPA (still fast on modern GPUs)")
            print("="*80 + "\n")
            return False

    except subprocess.TimeoutExpired:
        print("\n" + "="*80)
        print("⚠️  Flash Attention installation timed out (non-critical)")
        print("   Model will fall back to SDPA")
        print("="*80 + "\n")
        return False
    except Exception as e:
        print(f"\n⚠️  Flash Attention installation error: {e}")
        print("   Model will fall back to SDPA (non-critical)\n")
        return False


def compile_mesh_inpaint_processor():
    """
    Compile the mesh_inpaint_processor C++ extension for texture inpainting.

    This enables vertex-aware UV texture inpainting for high-quality PBR textures.
    """
    print("\n" + "="*80)
    print("🔧 ComfyUI-MeshCraft: Checking mesh_inpaint_processor...")
    print("="*80 + "\n")

    # Check if already compiled
    current_dir = os.path.dirname(os.path.abspath(__file__))
    renderer_dir = os.path.join(current_dir, "nodes", "lib", "hy3dpaint", "DifferentiableRenderer")

    # Look for compiled .so file
    import glob
    so_files = glob.glob(os.path.join(renderer_dir, "mesh_inpaint_processor*.so"))

    if so_files:
        print(f"✅ mesh_inpaint_processor already compiled: {os.path.basename(so_files[0])}")
        return True

    print("   mesh_inpaint_processor not compiled, building now...")
    print("   This enables vertex-aware texture inpainting\n")

    # Ensure pybind11 is installed
    try:
        import pybind11
    except ImportError:
        print("   Installing pybind11...")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "pybind11"],
            stdout=sys.stdout,
            stderr=sys.stderr,
            timeout=60
        )
        if result.returncode != 0:
            print("⚠️  Failed to install pybind11, skipping mesh_inpaint_processor")
            return False

    # Ensure python3-dev is installed (Linux only)
    if sys.platform == "linux":
        import shutil
        if not shutil.which("python3-config"):
            print("   Installing python3-dev...")
            try:
                is_root = os.geteuid() == 0
                cmd_prefix = [] if is_root else ["sudo", "-n"]

                result = subprocess.run(
                    cmd_prefix + ["apt-get", "install", "-y", "python3-dev"],
                    stdout=sys.stdout,
                    stderr=sys.stderr,
                    timeout=120
                )
                if result.returncode != 0:
                    print("⚠️  Failed to install python3-dev, skipping compilation")
                    return False
            except Exception as e:
                print(f"⚠️  Could not install python3-dev: {e}")
                return False

    # Compile the extension
    compile_script = os.path.join(renderer_dir, "compile_mesh_painter.sh")
    if not os.path.exists(compile_script):
        print(f"⚠️  Compile script not found: {compile_script}")
        return False

    try:
        original_dir = os.getcwd()
        os.chdir(renderer_dir)

        result = subprocess.run(
            ["bash", "compile_mesh_painter.sh"],
            stdout=sys.stdout,
            stderr=sys.stderr,
            timeout=120
        )

        os.chdir(original_dir)

        if result.returncode == 0:
            # Verify compilation succeeded
            so_files = glob.glob(os.path.join(renderer_dir, "mesh_inpaint_processor*.so"))
            if so_files:
                print("\n" + "="*80)
                print("✅ mesh_inpaint_processor compiled successfully!")
                print(f"   Built: {os.path.basename(so_files[0])}")
                print("="*80 + "\n")
                return True
            else:
                print("\n⚠️  Compilation appeared to succeed but .so file not found")
                return False
        else:
            print("\n" + "="*80)
            print("⚠️  mesh_inpaint_processor compilation failed")
            print("   Texture inpainting may not work optimally")
            print("   Check the output above for details")
            print("="*80 + "\n")
            return False

    except subprocess.TimeoutExpired:
        print("⚠️  Compilation timed out")
        os.chdir(original_dir)
        return False
    except Exception as e:
        print(f"⚠️  Compilation error: {e}")
        os.chdir(original_dir)
        return False


def install_blender():
    """
    Install Blender if not already installed.

    Blender is required for high-quality RGB multiview rendering that matches
    Hunyuan3D-2.1 training data preprocessing.
    """
    import shutil

    # Check if Blender is already available
    blender_path = os.environ.get("BLENDER_PATH")
    if blender_path and os.path.exists(blender_path):
        print(f"✅ ComfyUI-MeshCraft: Blender found at {blender_path}")
        return True

    # Check common locations and PATH
    if shutil.which("blender"):
        print(f"✅ ComfyUI-MeshCraft: Blender found in PATH")
        return True

    common_paths = [
        "/usr/bin/blender",
        "/usr/local/bin/blender",
        "/Applications/Blender.app/Contents/MacOS/Blender",
    ]
    for path in common_paths:
        if os.path.exists(path):
            print(f"✅ ComfyUI-MeshCraft: Blender found at {path}")
            return True

    # Blender not found - attempt installation
    print("\n" + "="*80)
    print("🔧 ComfyUI-MeshCraft: Blender not found, attempting to install...")
    print("   Blender is needed for RGB multiview rendering (matching Hunyuan3D training)")
    print("="*80 + "\n")

    # Only auto-install on Linux (apt-based systems)
    if sys.platform != "linux":
        print("⚠️  ComfyUI-MeshCraft: Auto-install only supported on Linux")
        print("   Please install Blender manually from https://www.blender.org/download/")
        print("   Or set BLENDER_PATH environment variable")
        return False

    try:
        # Check if running as root or if sudo is available
        is_root = os.geteuid() == 0

        if not is_root and not shutil.which("sudo"):
            print("⚠️  ComfyUI-MeshCraft: sudo not available, cannot install Blender")
            print("   Please install manually: apt-get install blender")
            return False

        # Update package list
        cmd_prefix = [] if is_root else ["sudo", "-n"]  # -n = non-interactive

        print("   Updating package list...")
        result = subprocess.run(
            cmd_prefix + ["apt-get", "update"],
            stdout=sys.stdout,
            stderr=sys.stderr,
            timeout=120
        )

        if result.returncode != 0:
            print("⚠️  ComfyUI-MeshCraft: apt-get update failed")
            return False

        # Install Blender + required OpenGL and X11 libraries for headless rendering
        # OpenGL libraries (libopengl0, libglu1-mesa, libosmesa6) needed for PyMeshLab plugins
        print("   Installing Blender + OpenGL/X11 libraries (this may take a few minutes)...")
        result = subprocess.run(
            cmd_prefix + ["apt-get", "install", "-y", "blender", "libegl1", "libgl1", "libgomp1",
                         "libxrender1", "libxi6", "libxrandr2", "libxxf86vm1", "libxfixes3",
                         "libopengl0", "libglu1-mesa", "libosmesa6"],
            stdout=sys.stdout,
            stderr=sys.stderr,
            timeout=600  # 10 minute timeout
        )

        if result.returncode == 0:
            print("\n" + "="*80)
            print("✅ ComfyUI-MeshCraft: Blender installed successfully!")
            print("="*80 + "\n")
            return True
        else:
            print("\n" + "="*80)
            print("⚠️  ComfyUI-MeshCraft: Blender installation failed")
            print("   Please install manually: sudo apt-get install blender")
            print("="*80 + "\n")
            return False

    except subprocess.TimeoutExpired:
        print("⚠️  ComfyUI-MeshCraft: Blender installation timed out")
        return False
    except Exception as e:
        print(f"⚠️  ComfyUI-MeshCraft: Blender installation error: {e}")
        return False


def install_trellis_dependencies():
    """
    Install TRELLIS-specific dependencies for multi-view 3D generation.

    TRELLIS requires:
    - imageio with ffmpeg support (for video rendering)
    - einops (for tensor operations)
    - Optional: flash-attn or xformers (for performance)

    Heavy dependencies (spconv, kaolin, nvdiffrast) are left for manual installation
    as they require CUDA compilation and are version-specific.
    """
    print("\n" + "="*80)
    print("🎲 ComfyUI-MeshCraft: Checking TRELLIS dependencies...")
    print("="*80 + "\n")

    # List of pip-installable dependencies
    dependencies = [
        ("imageio", ">=2.0.0"),
        ("imageio-ffmpeg", None),  # For video export
        ("einops", ">=0.7.0"),
    ]

    all_installed = True

    for package, version_spec in dependencies:
        try:
            # Try to import the package
            __import__(package.replace("-", "_"))
            print(f"✅ {package} already installed")
        except ImportError:
            print(f"   Installing {package}{version_spec or ''}...")
            try:
                install_spec = f"{package}{version_spec}" if version_spec else package
                result = subprocess.run(
                    [sys.executable, "-m", "pip", "install", install_spec, "-q"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=120
                )

                if result.returncode == 0:
                    print(f"✅ {package} installed successfully")
                else:
                    print(f"⚠️  Failed to install {package}")
                    all_installed = False

            except Exception as e:
                print(f"⚠️  Error installing {package}: {e}")
                all_installed = False

    # Check for TRELLIS package (directory check only - DO NOT import torch!)
    trellis_path = os.path.join(os.path.dirname(__file__), "nodes", "lib", "trellis")
    trellis_pipeline_file = os.path.join(trellis_path, "pipelines", "trellis_image_to_3d.py")

    if os.path.exists(trellis_pipeline_file):
        print("✅ TRELLIS package found (local repository)")
    else:
        print("⚠️  TRELLIS package not found")
        print("   The TRELLIS submodule may not be initialized")
        print("   Run: git submodule update --init --recursive")
        all_installed = False

    # Inform about manual dependencies
    print("\n" + "-"*80)
    print("ℹ️  Optional TRELLIS dependencies (install manually for best performance):")
    print("   • flash-attn (10-20% faster): pip install flash-attn --no-build-isolation")
    print("   • xformers (alternative): pip install xformers")
    print("\n   Advanced (require compilation, version-specific):")
    print("   • spconv (sparse convolution): See https://github.com/traveller59/spconv")
    print("   • kaolin (3D ops): See https://github.com/NVIDIAGameWorks/kaolin")
    print("   • nvdiffrast (rendering): See https://github.com/NVlabs/nvdiffrast")
    print("-"*80 + "\n")

    if all_installed:
        print("✅ All basic TRELLIS dependencies installed!")
    else:
        print("⚠️  Some TRELLIS dependencies failed to install")
        print("   TRELLIS nodes may not work correctly")

    print("="*80 + "\n")

    return all_installed


# Run on import
if __name__ != "__main__":
    # Set up library paths first
    setup_torch_library_path()

    # Install Flash Attention for optimal diffusion performance
    # DISABLED: Takes 5-10 minutes to compile, non-critical (10-20% speedup)
    # To install manually: pip install flash-attn==2.8.2 --no-build-isolation
    # install_flash_attention()

    # Compile mesh inpaint processor for texture inpainting
    compile_mesh_inpaint_processor()

    # Install Blender if needed (for RGB multiview rendering)
    install_blender()

    # Compile CUDA extension if needed
    compile_cuda_extension()

    # Install TRELLIS dependencies
    install_trellis_dependencies()
