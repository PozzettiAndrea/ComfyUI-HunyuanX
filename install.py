"""
ComfyUI-MeshCraft Installation Script

This script runs when the node is installed via ComfyUI-Manager.
It performs all one-time setup operations:
- Installing Python dependencies
- Installing system packages (CUDA libs, Blender, etc.)
- Compiling CUDA/C++ extensions
- Setting up the environment

This is separated from prestartup_script.py to minimize startup time.
prestartup_script.py only does fast runtime checks and environment setup.
"""

import subprocess
import sys
import os
import site
import shutil
import glob


def print_header(message):
    """Print a formatted header message"""
    print("\n" + "="*80)
    print(message)
    print("="*80 + "\n")


def print_subheader(message):
    """Print a formatted subheader message"""
    print("\n" + "-"*80)
    print(message)
    print("-"*80 + "\n")


def check_and_fix_pytorch_version():
    """Check PyTorch version and automatically upgrade/downgrade if needed"""
    print_header("🔍 Checking PyTorch Version")

    try:
        import torch
        torch_version = torch.__version__.split('+')[0]  # "2.9.0+cu128" -> "2.9.0"
        cuda_version = torch.version.cuda  # e.g., "12.8"

        print(f"   Current: PyTorch {torch_version}, CUDA {cuda_version}")
        print(f"   Required: PyTorch 2.5.0-2.8.0 (for Kaolin compatibility)")

        # Parse version for comparison
        version_parts = [int(x) for x in torch_version.split('.')]
        major, minor, patch = version_parts[0], version_parts[1], version_parts[2] if len(version_parts) > 2 else 0

        # Check if version is in range 2.5.0 - 2.8.0
        is_valid = (major == 2 and 5 <= minor <= 8)

        if is_valid:
            print("✅ PyTorch version is compatible with Kaolin")
            return True

        # Version is wrong - need to fix it
        print(f"\n⚠️  PyTorch {torch_version} is NOT compatible with Kaolin")
        print("   Kaolin requires PyTorch 2.5.0 - 2.8.0")
        print("\n   Installing PyTorch 2.8.0 (recommended version)...\n")

        # Determine CUDA version for installation URL
        cuda_ver_short = cuda_version.replace('.', '') if cuda_version else '128'
        cuda_tag = f"cu{cuda_ver_short}"

        # Install PyTorch 2.8.0 (let pip auto-resolve compatible torchvision/torchaudio)
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install",
             "torch==2.8.0", "torchvision", "torchaudio",
             "--extra-index-url", f"https://download.pytorch.org/whl/{cuda_tag}",
             "--force-reinstall"],
            stdout=sys.stdout,
            stderr=sys.stderr,
            timeout=600
        )

        if result.returncode == 0:
            print("\n✅ PyTorch 2.8.0 installed successfully!")
            print("   Kaolin will now be able to install from prebuilt wheels")
            return True
        else:
            print("\n⚠️  PyTorch installation failed")
            print("   Please install manually:")
            print(f"   pip install torch==2.8.0 torchvision torchaudio \\")
            print(f"       --extra-index-url https://download.pytorch.org/whl/{cuda_tag} --force-reinstall")
            return False

    except ImportError:
        print("⚠️  PyTorch not installed!")
        print("   Installing PyTorch 2.8.0 + CUDA 12.8...\n")

        result = subprocess.run(
            [sys.executable, "-m", "pip", "install",
             "torch==2.8.0", "torchvision", "torchaudio",
             "--extra-index-url", "https://download.pytorch.org/whl/cu128"],
            stdout=sys.stdout,
            stderr=sys.stderr,
            timeout=600
        )

        if result.returncode == 0:
            print("\n✅ PyTorch 2.8.0 installed successfully!")
            return True
        else:
            print("\n⚠️  PyTorch installation failed")
            return False

    except Exception as e:
        print(f"\n⚠️  Error checking PyTorch: {e}")
        return False


def install_python_dependencies(current_dir):
    """Install Python dependencies from requirements.txt"""
    print_header("📥 Installing Python Dependencies")

    requirements_txt = os.path.join(current_dir, "requirements.txt")

    if not os.path.exists(requirements_txt):
        print("⚠️  requirements.txt not found, skipping")
        return True

    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", requirements_txt],
            stdout=sys.stdout,
            stderr=sys.stderr,
            timeout=300
        )

        if result.returncode == 0:
            print("✅ Python dependencies installed successfully")
            return True
        else:
            print("⚠️  Some dependencies may have failed to install")
            return False

    except Exception as e:
        print(f"⚠️  Error installing dependencies: {e}")
        return False


def install_trellis_dependencies():
    """Install TRELLIS-specific dependencies"""
    print_header("🎲 Installing TRELLIS Dependencies")

    dependencies = [
        ("imageio", ">=2.0.0"),
        ("imageio-ffmpeg", None),
        ("einops", ">=0.7.0"),
    ]

    all_installed = True

    for package, version_spec in dependencies:
        try:
            __import__(package.replace("-", "_"))
            print(f"✅ {package} already installed")
        except ImportError:
            print(f"   Installing {package}{version_spec or ''}...")
            try:
                install_spec = f"{package}{version_spec}" if version_spec else package
                result = subprocess.run(
                    [sys.executable, "-m", "pip", "install", install_spec],
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

    print_subheader("ℹ️  Optional TRELLIS dependencies (install manually for best performance):")
    print("   • flash-attn (10-20% faster): pip install flash-attn --no-build-isolation")
    print("   • xformers (alternative): pip install xformers")
    print("\n   Advanced (require compilation, version-specific):")
    print("   • spconv (sparse convolution): See https://github.com/traveller59/spconv")
    print("   • kaolin (3D ops): See https://github.com/NVIDIAGameWorks/kaolin")
    print("   • nvdiffrast (rendering): See https://github.com/NVlabs/nvdiffrast")

    return all_installed


def check_cuda_libraries():
    """Check if required CUDA development libraries are installed"""
    cuda_home = os.environ.get('CUDA_HOME', '/usr/local/cuda')

    # Try common CUDA locations
    if not os.path.exists(cuda_home):
        for path in ['/usr/local/cuda-12.8', '/usr/local/cuda-12', '/usr/local/cuda']:
            if os.path.exists(path):
                cuda_home = path
                break

    required_headers = ['cusparse.h', 'cublas_v2.h', 'cufft.h']
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
    """Install CUDA development libraries if missing"""
    print_header("📦 Installing CUDA Development Libraries")

    if sys.platform != "linux":
        print("⚠️  Auto-installation only supported on Linux")
        print("   Please install CUDA toolkit manually from https://developer.nvidia.com/cuda-downloads")
        return False

    try:
        is_root = os.geteuid() == 0
    except AttributeError:
        print("⚠️  Cannot determine user privileges")
        return False

    if not is_root and not shutil.which("sudo"):
        print("⚠️  sudo not available, cannot auto-install")
        print("   Please install manually: sudo apt-get install cuda-libraries-dev-12-8")
        return False

    cmd_prefix = [] if is_root else ["sudo", "-n"]

    try:
        print("   Updating package list...")
        subprocess.run(
            cmd_prefix + ["apt-get", "update", "-qq"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=120
        )

        print("   Installing CUDA libraries (this may take 1-2 minutes)...")
        result = subprocess.run(
            cmd_prefix + ["apt-get", "install", "-y", "cuda-libraries-dev-12-8"],
            stdout=sys.stdout,
            stderr=sys.stderr,
            timeout=300
        )

        if result.returncode == 0:
            print("✅ CUDA libraries installed successfully")
            return True
        else:
            print("⚠️  Installation failed")
            return False

    except Exception as e:
        print(f"⚠️  Installation error: {e}")
        return False


def install_system_packages():
    """Install system packages (python3-dev for pybind11 compilation)"""
    print_header("📦 Installing System Packages")

    if sys.platform != "linux":
        print("ℹ️  Skipping system package installation (Linux only)")
        return True  # Non-critical on other platforms

    # Check if python3-dev is needed (for pybind11)
    if shutil.which("python3-config"):
        print("✅ python3-dev already installed")
        return True

    try:
        is_root = os.geteuid() == 0
        if not is_root and not shutil.which("sudo"):
            print("⚠️  sudo not available, cannot install python3-dev")
            return False

        cmd_prefix = [] if is_root else ["sudo", "-n"]

        print("   Installing python3-dev...")
        result = subprocess.run(
            cmd_prefix + ["apt-get", "install", "-y", "python3-dev"],
            stdout=sys.stdout,
            stderr=sys.stderr,
            timeout=120
        )

        if result.returncode == 0:
            print("✅ python3-dev installed successfully")
            return True
        else:
            print("⚠️  Failed to install python3-dev")
            return False

    except Exception as e:
        print(f"⚠️  Error installing python3-dev: {e}")
        return False


def compile_cuda_extension(current_dir):
    """Compile the custom rasterizer CUDA extension"""
    print_header("🔧 Compiling CUDA Rasterizer Extension")

    # Check if already compiled
    site_packages = site.getsitepackages()[0] if site.getsitepackages() else None
    if site_packages:
        pkg_dir = os.path.join(site_packages, "custom_rasterizer")
        if os.path.exists(pkg_dir):
            print("✅ custom_rasterizer already compiled")
            return True

    # Check CUDA libraries
    libs_available, missing = check_cuda_libraries()
    if not libs_available:
        print(f"⚠️  Missing CUDA development libraries: {', '.join(missing)}")
        if not install_cuda_libraries():
            print("⚠️  Skipping CUDA extension compilation")
            print("   Basic 3D mesh generation will still work")
            return False

    # Get path to setup.py
    rasterizer_dir = os.path.join(current_dir, "nodes", "lib", "hy3dpaint", "custom_rasterizer")
    setup_py = os.path.join(rasterizer_dir, "setup.py")

    if not os.path.exists(setup_py):
        print(f"⚠️  setup.py not found at {setup_py}")
        return False

    print("   Compiling (this may take 1-2 minutes)...")

    try:
        original_dir = os.getcwd()
        os.chdir(rasterizer_dir)

        result = subprocess.run(
            [sys.executable, "setup.py", "install"],
            stdout=sys.stdout,
            stderr=sys.stderr,
            timeout=300
        )

        os.chdir(original_dir)

        if result.returncode == 0:
            print("✅ CUDA extension compiled successfully")
            return True
        else:
            print("⚠️  Compilation failed")
            return False

    except Exception as e:
        print(f"⚠️  Compilation error: {e}")
        os.chdir(original_dir)
        return False


def compile_mesh_inpaint_processor(current_dir):
    """Compile the mesh_inpaint_processor C++ extension"""
    print_header("🔧 Compiling Mesh Inpaint Processor")

    renderer_dir = os.path.join(current_dir, "nodes", "lib", "hy3dpaint", "DifferentiableRenderer")

    # Check if already compiled
    so_files = glob.glob(os.path.join(renderer_dir, "mesh_inpaint_processor*.so"))
    if so_files:
        print(f"✅ mesh_inpaint_processor already compiled: {os.path.basename(so_files[0])}")
        return True

    # Ensure pybind11 is installed
    try:
        import pybind11
        print("✅ pybind11 already installed")
    except ImportError:
        print("   Installing pybind11...")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "pybind11"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=60
        )
        if result.returncode != 0:
            print("⚠️  Failed to install pybind11")
            return False

    # Ensure python3-dev is installed
    if not install_system_packages():
        print("⚠️  Cannot compile without python3-dev")
        return False

    # Compile the extension
    compile_script = os.path.join(renderer_dir, "compile_mesh_painter.sh")
    if not os.path.exists(compile_script):
        print(f"⚠️  Compile script not found: {compile_script}")
        return False

    print("   Compiling (this may take 30 seconds)...")

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
            so_files = glob.glob(os.path.join(renderer_dir, "mesh_inpaint_processor*.so"))
            if so_files:
                print(f"✅ mesh_inpaint_processor compiled successfully: {os.path.basename(so_files[0])}")
                return True
            else:
                print("⚠️  Compilation appeared to succeed but .so file not found")
                return False
        else:
            print("⚠️  Compilation failed")
            return False

    except Exception as e:
        print(f"⚠️  Compilation error: {e}")
        os.chdir(original_dir)
        return False


def install_kaolin():
    """Install NVIDIA Kaolin for 3D deep learning operations"""
    print_header("🎮 Installing NVIDIA Kaolin")

    # Check if already installed (and it's not the placeholder)
    try:
        import kaolin
        # Try to access __version__ - placeholder wheel will fail here
        version = kaolin.__version__
        print(f"✅ Kaolin {version} already installed")
        return True
    except (ImportError, AttributeError):
        # Either not installed or it's the placeholder wheel
        pass

    print("   Kaolin is required for TRELLIS FlexiCubes mesh extraction")
    print("   Installing from NVIDIA prebuilt wheels...\n")

    try:
        # Detect PyTorch and CUDA versions
        import torch
        torch_version = torch.__version__.split('+')[0]  # e.g., "2.9.0"
        cuda_version = torch.version.cuda  # e.g., "12.8"

        print(f"   Detected: PyTorch {torch_version}, CUDA {cuda_version}")

        # Format versions for kaolin wheel URL
        # torch: "2.9.0" -> "2.9.0"
        # cuda: "12.8" -> "cu128"
        cuda_ver_short = cuda_version.replace('.', '')  # "12.8" -> "128"
        cuda_tag = f"cu{cuda_ver_short}"

        # Kaolin version to install
        kaolin_version = "0.18.0"

        # List of torch versions to try (in order of preference)
        # Start with exact match, then try compatible versions
        torch_versions_to_try = [
            torch_version,  # Exact match (e.g., "2.9.0")
            ".".join(torch_version.split(".")[:2]) + ".0",  # Try .0 patch (e.g., "2.9.0")
            "2.8.0",  # Known compatible version
            "2.7.1",  # Fallback
        ]

        # Remove duplicates while preserving order
        seen = set()
        torch_versions_to_try = [x for x in torch_versions_to_try if not (x in seen or seen.add(x))]

        # Try each torch version
        for torch_ver in torch_versions_to_try:
            wheel_url = f"https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-{torch_ver}_{cuda_tag}.html"

            print(f"\n   Trying: kaolin {kaolin_version} for torch {torch_ver} + {cuda_tag}...")
            print(f"   Wheel URL: {wheel_url}")

            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", f"kaolin=={kaolin_version}",
                 "-f", wheel_url, "--no-cache-dir"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=600  # 10 minute timeout per attempt
            )

            if result.returncode == 0:
                # Verify installation worked (not placeholder)
                try:
                    import kaolin
                    version = kaolin.__version__
                    print(f"\n✅ Kaolin {version} installed successfully!")
                    return True
                except (ImportError, AttributeError):
                    print("   ⚠️  Installed placeholder wheel, trying next version...")
                    continue
            else:
                stderr = result.stderr.decode('utf-8', errors='ignore')
                if "Could not find" in stderr or "404" in stderr:
                    print(f"   ⚠️  Wheel not available for this combination, trying next...")
                else:
                    print(f"   ⚠️  Installation failed: {stderr[:200]}")

        # All prebuilt wheels failed, try from source as last resort
        print("\n⚠️  All prebuilt wheels failed. Attempting to install from source...")
        print("   This may take 10-30 minutes and requires build tools.\n")

        result = subprocess.run(
            [sys.executable, "-m", "pip", "install",
             "git+https://github.com/NVIDIAGameWorks/kaolin.git@v0.18.0"],
            stdout=sys.stdout,
            stderr=sys.stderr,
            timeout=1800  # 30 minute timeout for source build
        )

        if result.returncode == 0:
            print("\n✅ Kaolin installed from source successfully!")
            return True
        else:
            print("\n⚠️  Kaolin installation from source failed")
            print("\nManual installation instructions:")
            print("   1. Visit: https://kaolin.readthedocs.io/en/latest/notes/installation.html")
            print("   2. Check compatibility table for your PyTorch/CUDA versions")
            print("   3. Install with:")
            print(f"      pip install kaolin==0.18.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-{torch_version}_{cuda_tag}.html")
            return False

    except subprocess.TimeoutExpired:
        print("\n⚠️  Kaolin installation timed out")
        return False
    except Exception as e:
        print(f"\n⚠️  Kaolin installation error: {e}")
        import traceback
        traceback.print_exc()
        return False


def install_blender():
    """Install Blender for RGB multiview rendering"""
    print_header("🔧 Installing Blender")

    # Check if already installed
    blender_path = os.environ.get("BLENDER_PATH")
    if blender_path and os.path.exists(blender_path):
        print(f"✅ Blender found at {blender_path}")
        return True

    if shutil.which("blender"):
        print("✅ Blender found in PATH")
        return True

    common_paths = [
        "/usr/bin/blender",
        "/usr/local/bin/blender",
        "/Applications/Blender.app/Contents/MacOS/Blender",
    ]
    for path in common_paths:
        if os.path.exists(path):
            print(f"✅ Blender found at {path}")
            return True

    # Attempt installation
    if sys.platform != "linux":
        print("⚠️  Auto-installation only supported on Linux")
        print("   Please install Blender manually from https://www.blender.org/download/")
        return False

    try:
        is_root = os.geteuid() == 0

        if not is_root and not shutil.which("sudo"):
            print("⚠️  sudo not available, cannot install Blender")
            print("   Please install manually: sudo apt-get install blender")
            return False

        cmd_prefix = [] if is_root else ["sudo", "-n"]

        print("   Updating package list...")
        subprocess.run(
            cmd_prefix + ["apt-get", "update", "-qq"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=120
        )

        print("   Installing Blender + OpenGL/X11 libraries (this may take a few minutes)...")
        result = subprocess.run(
            cmd_prefix + ["apt-get", "install", "-y", "blender", "libegl1", "libgl1",
                         "libgomp1", "libxrender1", "libxi6", "libxrandr2", "libxxf86vm1",
                         "libxfixes3", "libopengl0", "libglu1-mesa", "libosmesa6"],
            stdout=sys.stdout,
            stderr=sys.stderr,
            timeout=600
        )

        if result.returncode == 0:
            print("✅ Blender installed successfully")
            return True
        else:
            print("⚠️  Blender installation failed")
            return False

    except Exception as e:
        print(f"⚠️  Blender installation error: {e}")
        return False


def main():
    """Main installation function"""
    print_header("🎨 ComfyUI-MeshCraft Installation")
    print("This will install all dependencies and compile extensions.")
    print("This process may take 5-10 minutes depending on your system.\n")

    # Check and fix PyTorch version FIRST (required for Kaolin)
    if not check_and_fix_pytorch_version():
        print("\n" + "="*80)
        print("⚠️  Installation cannot continue without compatible PyTorch version")
        print("   Please fix PyTorch installation and run this script again.")
        print("="*80 + "\n")
        sys.exit(1)

    # Get MeshCraft root directory (go up from scripts/)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    current_dir = os.path.dirname(script_dir)  # Go up to MeshCraft root

    results = {}

    # Install Python dependencies
    results['python_deps'] = install_python_dependencies(current_dir)

    # Install TRELLIS dependencies
    results['trellis_deps'] = install_trellis_dependencies()

    # Install system packages (python3-dev)
    results['system_packages'] = install_system_packages()

    # Compile CUDA extension
    results['cuda_extension'] = compile_cuda_extension(current_dir)

    # Compile mesh inpaint processor
    results['mesh_inpaint'] = compile_mesh_inpaint_processor(current_dir)

    # Install Blender
    results['blender'] = install_blender()

    # Install Kaolin (required for TRELLIS)
    results['kaolin'] = install_kaolin()

    # Print summary
    print_header("📊 Installation Summary")

    status_emoji = {True: "✅", False: "⚠️"}

    print(f"{status_emoji[results['python_deps']]} Python dependencies")
    print(f"{status_emoji[results['trellis_deps']]} TRELLIS dependencies")
    print(f"{status_emoji[results['system_packages']]} System packages")
    print(f"{status_emoji[results['cuda_extension']]} CUDA rasterizer extension")
    print(f"{status_emoji[results['mesh_inpaint']]} Mesh inpaint processor")
    print(f"{status_emoji[results['blender']]} Blender")
    print(f"{status_emoji[results['kaolin']]} Kaolin (NVIDIA 3D library)")

    all_critical = results['python_deps'] and results['trellis_deps'] and results['kaolin']

    if all_critical:
        print_header("✅ ComfyUI-MeshCraft Installation Complete!")
        print("Please restart ComfyUI to load the nodes.")

        if not all(results.values()):
            print("\nNote: Some optional components failed to install.")
            print("Basic functionality will work, but some features may be limited.")
    else:
        print_header("⚠️  Installation Completed with Errors")
        print("Some critical dependencies failed to install.")
        print("Please check the output above for details.")
        print("\nYou may need to install some components manually.")


if __name__ == "__main__":
    main()
