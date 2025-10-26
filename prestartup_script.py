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
            capture_output=True,
            timeout=120,
            text=True
        )

        # Install cuda-libraries-dev-12-8
        print("   Installing CUDA libraries (this may take 1-2 minutes)...")
        result = subprocess.run(
            cmd_prefix + ["apt-get", "install", "-y", "cuda-libraries-dev-12-8"],
            capture_output=True,
            timeout=300,
            text=True
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
    rasterizer_dir = os.path.join(current_dir, "hy3dpaint", "custom_rasterizer")
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
            capture_output=True,
            text=True,
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
            print("   STDOUT:", result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)
            print("   STDERR:", result.stderr[-500:] if len(result.stderr) > 500 else result.stderr)
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
            capture_output=True,
            timeout=120
        )

        if result.returncode != 0:
            print("⚠️  ComfyUI-MeshCraft: apt-get update failed")
            return False

        # Install Blender + required OpenGL libraries for headless rendering
        print("   Installing Blender + OpenGL libraries (this may take a few minutes)...")
        result = subprocess.run(
            cmd_prefix + ["apt-get", "install", "-y", "blender", "libegl1", "libgl1", "libgomp1"],
            capture_output=True,
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


# Run on import
if __name__ != "__main__":
    # Set up library paths first
    setup_torch_library_path()

    # Install Blender if needed (for RGB multiview rendering)
    install_blender()

    # Compile CUDA extension if needed
    compile_cuda_extension()
