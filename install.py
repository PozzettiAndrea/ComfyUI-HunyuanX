"""
ComfyUI-HunyuanX Installation Script

Called by ComfyUI Manager during installation/update.
Installs dependencies and optionally compiles CUDA extensions.
"""
import os
import subprocess
import sys


def install_requirements():
    """
    Install dependencies from requirements.txt.
    """
    print("[ComfyUI-HunyuanX] Installing requirements.txt dependencies...")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    requirements_path = os.path.join(script_dir, "requirements.txt")

    if not os.path.exists(requirements_path):
        print("[ComfyUI-HunyuanX] ⚠️  requirements.txt not found, skipping")
        return False

    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", requirements_path],
            capture_output=True,
            text=True,
            timeout=600
        )

        if result.returncode == 0:
            print("[ComfyUI-HunyuanX] ✅ Requirements installed successfully")
            return True
        else:
            print("[ComfyUI-HunyuanX] ⚠️  Requirements installation had issues")
            if result.stderr:
                print(f"[ComfyUI-HunyuanX] Error details: {result.stderr[:500]}")
            return False

    except Exception as e:
        print(f"[ComfyUI-HunyuanX] ⚠️  Requirements installation error: {e}")
        return False


def get_torch_cuda_version():
    """
    Detect installed torch and CUDA versions.
    Returns tuple: (torch_version, cuda_version) or (None, None) if not found.
    """
    try:
        import torch
        torch_version = torch.__version__.split('+')[0]  # e.g., "2.9.0"
        cuda_version = torch.version.cuda  # e.g., "12.8"

        # Extract major.minor for torch (e.g., "2.9.0" -> "2.9")
        torch_major_minor = '.'.join(torch_version.split('.')[:2])

        # Extract major minor for CUDA and remove dots (e.g., "12.8" -> "128")
        cuda_compact = cuda_version.replace('.', '') if cuda_version else None

        return torch_major_minor, cuda_compact
    except Exception as e:
        print(f"[ComfyUI-HunyuanX] Could not detect torch/CUDA: {e}")
        return None, None


def get_python_version():
    """
    Get Python version in cpXX format (e.g., "cp310" for Python 3.10).
    """
    return f"cp{sys.version_info.major}{sys.version_info.minor}"


def check_cuda_toolkit_available():
    """
    Check if CUDA toolkit (nvcc compiler) is available.
    Returns: (is_available, cuda_home_path)
    """
    # Check for nvcc command
    try:
        result = subprocess.run(
            ["nvcc", "--version"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            # nvcc found, try to get CUDA_HOME
            cuda_home = os.environ.get('CUDA_HOME')
            if not cuda_home:
                # Try to infer from nvcc path
                nvcc_path_result = subprocess.run(
                    ["which", "nvcc"],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                if nvcc_path_result.returncode == 0:
                    nvcc_path = nvcc_path_result.stdout.strip()
                    # CUDA_HOME is typically two levels up from bin/nvcc
                    cuda_home = os.path.dirname(os.path.dirname(nvcc_path))

            return True, cuda_home
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        pass

    return False, None


def is_conda_environment():
    """
    Check if we're running in a conda environment.
    """
    return os.environ.get('CONDA_DEFAULT_ENV') is not None or os.environ.get('CONDA_PREFIX') is not None


def install_cuda_toolkit():
    """
    Automatically install CUDA toolkit matching PyTorch's CUDA version.
    Returns True if successful, False otherwise.
    """
    print("[ComfyUI-HunyuanX] CUDA toolkit (nvcc) not found - installing automatically...")

    # Get PyTorch's CUDA version
    _, cuda_version = get_torch_cuda_version()

    if not cuda_version:
        print("[ComfyUI-HunyuanX] ⚠️  Could not detect PyTorch CUDA version")
        print("[ComfyUI-HunyuanX] Cannot auto-install CUDA toolkit")
        return False

    # Convert cuda_version from compact format (e.g., "128") to dotted (e.g., "12.8")
    if len(cuda_version) == 3:
        cuda_major = cuda_version[0:2]
        cuda_minor = cuda_version[2]
        cuda_dotted = f"{cuda_major}.{cuda_minor}"
    elif len(cuda_version) == 2:
        cuda_major = cuda_version[0]
        cuda_minor = cuda_version[1]
        cuda_dotted = f"{cuda_major}.{cuda_minor}"
    else:
        print(f"[ComfyUI-HunyuanX] ⚠️  Unexpected CUDA version format: {cuda_version}")
        return False

    print(f"[ComfyUI-HunyuanX] Detected PyTorch CUDA version: {cuda_dotted}")

    # Check if we're in conda environment
    if is_conda_environment():
        print(f"[ComfyUI-HunyuanX] Installing CUDA toolkit {cuda_dotted} via conda...")
        print("[ComfyUI-HunyuanX] This may take a few minutes...")

        try:
            # Install cuda-toolkit and cuda-nvcc
            result = subprocess.run(
                ["conda", "install", "-y", "-c", "nvidia",
                 f"cuda-toolkit={cuda_dotted}",
                 f"cuda-nvcc={cuda_dotted}"],
                capture_output=True,
                text=True,
                timeout=600  # 10 minutes max
            )

            if result.returncode == 0:
                print("[ComfyUI-HunyuanX] ✅ CUDA toolkit installed successfully via conda")

                # Try to find CUDA_HOME from conda environment
                conda_prefix = os.environ.get('CONDA_PREFIX')
                if conda_prefix:
                    cuda_home = conda_prefix
                    os.environ['CUDA_HOME'] = cuda_home
                    print(f"[ComfyUI-HunyuanX] Set CUDA_HOME={cuda_home}")

                return True
            else:
                print("[ComfyUI-HunyuanX] ⚠️  Conda installation failed")
                if result.stderr:
                    print(f"[ComfyUI-HunyuanX] Error: {result.stderr[:500]}")
                return False

        except subprocess.TimeoutExpired:
            print("[ComfyUI-HunyuanX] ⚠️  CUDA toolkit installation timed out")
            return False
        except Exception as e:
            print(f"[ComfyUI-HunyuanX] ⚠️  CUDA toolkit installation error: {e}")
            return False
    else:
        # Not in conda - try pip installation
        print(f"[ComfyUI-HunyuanX] Not in conda environment, installing CUDA toolkit {cuda_dotted} via pip...")
        print("[ComfyUI-HunyuanX] This may take a few minutes...")

        try:
            # Extract major version for pip package naming (e.g., "12.8" -> "12")
            cuda_major = cuda_dotted.split('.')[0]

            # Install nvidia-cuda-nvcc and nvidia-cuda-runtime packages
            # These provide nvcc compiler and CUDA runtime
            packages = [
                f"nvidia-cuda-nvcc-cu{cuda_major}",
                f"nvidia-cuda-runtime-cu{cuda_major}"
            ]

            print(f"[ComfyUI-HunyuanX] Installing packages: {', '.join(packages)}")

            result = subprocess.run(
                [sys.executable, "-m", "pip", "install"] + packages,
                capture_output=True,
                text=True,
                timeout=600  # 10 minutes max
            )

            if result.returncode == 0:
                print("[ComfyUI-HunyuanX] ✅ CUDA toolkit installed successfully via pip")

                # Find CUDA_HOME from pip installation
                # nvcc is typically in site-packages/nvidia/cuda_nvcc/bin/nvcc
                try:
                    import site
                    site_packages = site.getsitepackages()
                    cuda_home = None

                    for sp in site_packages:
                        nvcc_path = os.path.join(sp, "nvidia", "cuda_nvcc")
                        if os.path.exists(nvcc_path):
                            cuda_home = nvcc_path
                            break

                    if cuda_home:
                        os.environ['CUDA_HOME'] = cuda_home
                        # Add nvcc to PATH
                        nvcc_bin = os.path.join(cuda_home, "bin")
                        if os.path.exists(nvcc_bin):
                            os.environ['PATH'] = nvcc_bin + os.pathsep + os.environ.get('PATH', '')
                        print(f"[ComfyUI-HunyuanX] Set CUDA_HOME={cuda_home}")
                    else:
                        print("[ComfyUI-HunyuanX] ⚠️  Warning: Could not locate CUDA_HOME in pip packages")

                except Exception as e:
                    print(f"[ComfyUI-HunyuanX] ⚠️  Error setting CUDA_HOME: {e}")

                return True
            else:
                print("[ComfyUI-HunyuanX] ⚠️  Pip installation failed")
                if result.stderr:
                    print(f"[ComfyUI-HunyuanX] Error: {result.stderr[:500]}")

                # Fall back to manual instructions
                print("[ComfyUI-HunyuanX] ⚠️  Could not auto-install CUDA toolkit")
                print("[ComfyUI-HunyuanX] Please install CUDA toolkit manually:")
                print(f"[ComfyUI-HunyuanX]   1. Download CUDA {cuda_dotted} from: https://developer.nvidia.com/cuda-downloads")
                print(f"[ComfyUI-HunyuanX]   2. Or install via pip: pip install nvidia-cuda-nvcc-cu{cuda_major}")
                print("[ComfyUI-HunyuanX]   3. Set CUDA_HOME environment variable")
                print("[ComfyUI-HunyuanX]   4. Re-run this installation script")
                return False

        except subprocess.TimeoutExpired:
            print("[ComfyUI-HunyuanX] ⚠️  CUDA toolkit installation timed out")
            return False
        except Exception as e:
            print(f"[ComfyUI-HunyuanX] ⚠️  CUDA toolkit installation error: {e}")

            # Fall back to manual instructions
            print("[ComfyUI-HunyuanX] Please install CUDA toolkit manually:")
            print(f"[ComfyUI-HunyuanX]   1. Download CUDA {cuda_dotted} from: https://developer.nvidia.com/cuda-downloads")
            print(f"[ComfyUI-HunyuanX]   2. Or install via pip: pip install nvidia-cuda-nvcc-cu{cuda_major}")
            return False


def try_install_flash_attn():
    """
    Attempt to install flash_attn (optional dependency for 10-20% faster inference).
    Tries PyPI wheels first, then custom prebuilt wheels. Never compiles from source.
    """
    print("[ComfyUI-HunyuanX] Checking for flash_attn...")

    # Check if already installed
    try:
        import flash_attn
        print("[ComfyUI-HunyuanX] ✅ flash_attn already installed")
        return True
    except ImportError:
        pass

    # Detect versions
    torch_version, cuda_version = get_torch_cuda_version()
    py_version = get_python_version()

    print(f"[ComfyUI-HunyuanX] Detected: Python {py_version}, Torch {torch_version}, CUDA {cuda_version}")

    # Step 1: Try PyPI with uv pip (wheels only, no compilation)
    print("[ComfyUI-HunyuanX] Step 1: Trying PyPI wheels with uv pip...")
    try:
        result = subprocess.run(
            [sys.executable, "-m", "uv", "pip", "install", "flash-attn", "--no-build"],
            capture_output=True,
            text=True,
            timeout=120
        )

        if result.returncode == 0:
            print("[ComfyUI-HunyuanX] ✅ flash_attn installed from PyPI")
            print("[ComfyUI-HunyuanX] Hunyuan 3D will use faster inference (10-20% speedup)")
            return True
        else:
            print("[ComfyUI-HunyuanX] ⚠️  No compatible wheel found on PyPI")
            if "ERROR" in result.stderr:
                # Extract just the error line, not full traceback
                error_lines = [line for line in result.stderr.split('\n') if 'ERROR' in line]
                if error_lines:
                    print(f"[ComfyUI-HunyuanX] {error_lines[0][:150]}")
    except Exception as e:
        print(f"[ComfyUI-HunyuanX] ⚠️  PyPI installation error: {e}")

    # Step 2: Try custom prebuilt wheels with multiple versions
    if torch_version and cuda_version:
        print("[ComfyUI-HunyuanX] Step 2: Trying custom prebuilt wheels...")

        # List of versions to try (newest first)
        versions_to_try = [
            ("2.8.3", "v0.4.24"),
            ("2.8.1", "v0.4.22"),
            ("2.7.0", "v0.4.20"),
            ("2.6.3", "v0.4.18"),
        ]

        for flash_version, release_tag in versions_to_try:
            wheel_url = (
                f"https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/{release_tag}/"
                f"flash_attn-{flash_version}%2Bcu{cuda_version}torch{torch_version}-{py_version}-{py_version}-linux_x86_64.whl"
            )

            print(f"[ComfyUI-HunyuanX] Trying flash_attn {flash_version}...")

            try:
                result = subprocess.run(
                    [sys.executable, "-m", "pip", "install", wheel_url],
                    capture_output=True,
                    text=True,
                    timeout=120
                )

                if result.returncode == 0:
                    print(f"[ComfyUI-HunyuanX] ✅ flash_attn {flash_version} installed successfully")
                    print("[ComfyUI-HunyuanX] Hunyuan 3D will use faster inference (10-20% speedup)")
                    return True
                else:
                    # Check if it's a 404 (not found) - try next version
                    if "404" in result.stderr or "Not Found" in result.stderr:
                        print(f"[ComfyUI-HunyuanX] Wheel not found for {flash_version}, trying next...")
                        continue
                    else:
                        # Other error - log and try next
                        print(f"[ComfyUI-HunyuanX] Installation failed: {result.stderr[:150]}")
                        continue
            except Exception as e:
                print(f"[ComfyUI-HunyuanX] ⚠️  Error with {flash_version}: {e}")
                continue

    # Step 3: Skip gracefully if no wheels found
    print("[ComfyUI-HunyuanX] ⚠️  No compatible flash_attn wheel found")
    print("[ComfyUI-HunyuanX] Continuing without flash_attn (nodes will still work, just ~10-20% slower)")
    print("[ComfyUI-HunyuanX] To manually install, try: uv pip install flash-attn --no-build")
    return False


def create_custom_rasterizer_pyproject():
    """
    Create pyproject.toml for custom_rasterizer to declare torch as build dependency.
    This ensures torch is available during the build process.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pyproject_path = os.path.join(script_dir, "nodes", "lib", "hy3dpaint", "custom_rasterizer", "pyproject.toml")

    # Only create if it doesn't exist
    if os.path.exists(pyproject_path):
        return True

    try:
        pyproject_content = """[build-system]
requires = ["setuptools", "wheel", "torch"]
build-backend = "setuptools.build_meta"

[project]
name = "custom_rasterizer"
version = "0.1"
description = "Custom CUDA rasterizer for texture generation"
requires-python = ">=3.7"
dependencies = []
"""
        with open(pyproject_path, 'w') as f:
            f.write(pyproject_content)

        print("[ComfyUI-HunyuanX] ✅ Created pyproject.toml for custom_rasterizer")
        return True

    except Exception as e:
        print(f"[ComfyUI-HunyuanX] ⚠️  Error creating pyproject.toml: {e}")
        return False


def patch_custom_rasterizer_setup():
    """
    Patch custom_rasterizer/setup.py to add compiler flags for compatibility.
    This fixes GCC 14+ compatibility issues with PyTorch CUDA extensions.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    setup_path = os.path.join(script_dir, "nodes", "lib", "hy3dpaint", "custom_rasterizer", "setup.py")

    if not os.path.exists(setup_path):
        return False

    try:
        with open(setup_path, 'r') as f:
            content = f.read()

        # Check if already patched with auto-detection logic
        if 'get_pytorch_abi' in content:
            print("[ComfyUI-HunyuanX] custom_rasterizer setup.py already patched with ABI auto-detection")
            return True

        # Check if already patched with the correct flags (must have -ccbin)
        if 'extra_compile_args' in content and '-ccbin' in content:
            print("[ComfyUI-HunyuanX] custom_rasterizer setup.py already patched")
            return True

        # Find the CUDAExtension definition and add compiler flags
        # Pattern without extra_compile_args
        original_pattern = '''custom_rasterizer_module = CUDAExtension(
    "custom_rasterizer_kernel",
    [
        "lib/custom_rasterizer_kernel/rasterizer.cpp",
        "lib/custom_rasterizer_kernel/grid_neighbor.cpp",
        "lib/custom_rasterizer_kernel/rasterizer_gpu.cu",
    ],
)'''

        # Pattern with old extra_compile_args (without -ccbin)
        old_patched_pattern1 = '''custom_rasterizer_module = CUDAExtension(
    "custom_rasterizer_kernel",
    [
        "lib/custom_rasterizer_kernel/rasterizer.cpp",
        "lib/custom_rasterizer_kernel/grid_neighbor.cpp",
        "lib/custom_rasterizer_kernel/rasterizer_gpu.cu",
    ],
    extra_compile_args={
        'cxx': ['-D_GLIBCXX_USE_CXX11_ABI=0'],
        'nvcc': ['-D_GLIBCXX_USE_CXX11_ABI=0']
    }
)'''

        # Pattern with -std=c++17 (needs -ccbin instead)
        old_patched_pattern2 = '''custom_rasterizer_module = CUDAExtension(
    "custom_rasterizer_kernel",
    [
        "lib/custom_rasterizer_kernel/rasterizer.cpp",
        "lib/custom_rasterizer_kernel/grid_neighbor.cpp",
        "lib/custom_rasterizer_kernel/rasterizer_gpu.cu",
    ],
    extra_compile_args={
        'cxx': ['-D_GLIBCXX_USE_CXX11_ABI=0', '-std=c++17'],
        'nvcc': ['-D_GLIBCXX_USE_CXX11_ABI=0', '-std=c++17']
    }
)'''

        # New patched pattern with compiler compatibility flags
        # Use system g++ if available (typically older and more compatible than conda's GCC 14+)
        import shutil
        system_gxx = shutil.which('g++', path='/usr/bin:/bin')

        if system_gxx:
            new_patched_pattern = f'''custom_rasterizer_module = CUDAExtension(
    "custom_rasterizer_kernel",
    [
        "lib/custom_rasterizer_kernel/rasterizer.cpp",
        "lib/custom_rasterizer_kernel/grid_neighbor.cpp",
        "lib/custom_rasterizer_kernel/rasterizer_gpu.cu",
    ],
    extra_compile_args={{
        'cxx': ['-D_GLIBCXX_USE_CXX11_ABI=0'],
        'nvcc': ['-D_GLIBCXX_USE_CXX11_ABI=0', '-ccbin', '{system_gxx}']
    }}
)'''
        else:
            # Fallback if no system g++ found
            new_patched_pattern = '''custom_rasterizer_module = CUDAExtension(
    "custom_rasterizer_kernel",
    [
        "lib/custom_rasterizer_kernel/rasterizer.cpp",
        "lib/custom_rasterizer_kernel/grid_neighbor.cpp",
        "lib/custom_rasterizer_kernel/rasterizer_gpu.cu",
    ],
    extra_compile_args={
        'cxx': ['-D_GLIBCXX_USE_CXX11_ABI=0'],
        'nvcc': ['-D_GLIBCXX_USE_CXX11_ABI=0']
    }
)'''

        # Try to replace original pattern or old patched patterns
        if original_pattern in content:
            content = content.replace(original_pattern, new_patched_pattern)
            with open(setup_path, 'w') as f:
                f.write(content)
            print("[ComfyUI-HunyuanX] ✅ Patched custom_rasterizer setup.py for compatibility")
            return True
        elif old_patched_pattern1 in content:
            content = content.replace(old_patched_pattern1, new_patched_pattern)
            with open(setup_path, 'w') as f:
                f.write(content)
            print("[ComfyUI-HunyuanX] ✅ Updated custom_rasterizer setup.py with improved flags")
            return True
        elif old_patched_pattern2 in content:
            content = content.replace(old_patched_pattern2, new_patched_pattern)
            with open(setup_path, 'w') as f:
                f.write(content)
            print("[ComfyUI-HunyuanX] ✅ Updated custom_rasterizer setup.py with -ccbin flag")
            return True
        else:
            print("[ComfyUI-HunyuanX] ⚠️  Could not find expected pattern in setup.py")
            return False

    except Exception as e:
        print(f"[ComfyUI-HunyuanX] ⚠️  Error patching setup.py: {e}")
        return False


def compile_cuda_extensions():
    """
    Compile Hunyuan-specific CUDA extensions:
    - custom_rasterizer (for texture generation)
    - mesh_inpaint_processor (for texture inpainting)
    """
    print("\n[ComfyUI-HunyuanX] Compiling CUDA extensions...")
    print("[ComfyUI-HunyuanX] This may take 2-5 minutes...\n")

    # Check if CUDA toolkit (nvcc) is available
    print("[ComfyUI-HunyuanX] Checking for CUDA toolkit...")
    cuda_available, cuda_home = check_cuda_toolkit_available()

    if not cuda_available:
        print("[ComfyUI-HunyuanX] ⚠️  CUDA toolkit (nvcc compiler) not found")

        # Try to install CUDA toolkit automatically
        cuda_installed = install_cuda_toolkit()

        if not cuda_installed:
            print("[ComfyUI-HunyuanX] ⚠️  Cannot compile CUDA extensions without CUDA toolkit")
            print("[ComfyUI-HunyuanX] Skipping CUDA extension compilation")
            print("[ComfyUI-HunyuanX] Some features may be limited (texture generation, etc.)")
            return

        # Re-check after installation
        cuda_available, cuda_home = check_cuda_toolkit_available()
        if not cuda_available:
            print("[ComfyUI-HunyuanX] ⚠️  CUDA toolkit installation succeeded but nvcc still not found")
            print("[ComfyUI-HunyuanX] Skipping CUDA extension compilation")
            return

    # Set CUDA_HOME if we found it
    if cuda_home and not os.environ.get('CUDA_HOME'):
        os.environ['CUDA_HOME'] = cuda_home
        print(f"[ComfyUI-HunyuanX] Set CUDA_HOME={cuda_home}")

    print(f"[ComfyUI-HunyuanX] ✅ CUDA toolkit found (CUDA_HOME={os.environ.get('CUDA_HOME', 'not set')})")

    # Install ninja for faster CUDA compilation
    print("[ComfyUI-HunyuanX] Checking for ninja build system...")
    try:
        import ninja
        print("[ComfyUI-HunyuanX] ✅ Ninja already installed")
    except ImportError:
        print("[ComfyUI-HunyuanX] Installing ninja for faster compilation...")
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "ninja"],
                capture_output=True,
                text=True,
                timeout=60
            )
            if result.returncode == 0:
                print("[ComfyUI-HunyuanX] ✅ Ninja installed successfully")
            else:
                print("[ComfyUI-HunyuanX] ⚠️  Ninja installation failed (will use slower distutils)")
        except Exception as e:
            print(f"[ComfyUI-HunyuanX] ⚠️  Could not install ninja: {e}")

    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Prepare custom_rasterizer for compilation
    create_custom_rasterizer_pyproject()
    patch_custom_rasterizer_setup()

    # 1. Compile custom_rasterizer
    print("[ComfyUI-HunyuanX] Compiling custom_rasterizer...")
    rasterizer_dir = os.path.join(script_dir, "nodes", "lib", "hy3dpaint", "custom_rasterizer")

    if os.path.exists(rasterizer_dir):
        try:
            # Use pip install with --no-build-isolation to access current env's torch
            # Force use of system compilers (avoids conda GCC 14+ header conflicts)
            env = os.environ.copy()
            import shutil
            system_gxx = shutil.which('g++', path='/usr/bin:/bin')
            system_gcc = shutil.which('gcc', path='/usr/bin:/bin')

            if system_gxx and system_gcc:
                env['CXX'] = system_gxx
                env['CC'] = system_gcc
                print(f"[ComfyUI-HunyuanX] Using system compilers for C/C++ files")

            # Set CUDA include path for conda-installed CUDA toolkit
            # Conda places headers in targets/x86_64-linux/include instead of include/
            cuda_home = os.environ.get('CUDA_HOME')
            if cuda_home:
                cuda_inc_path = os.path.join(cuda_home, 'targets', 'x86_64-linux', 'include')
                if os.path.exists(cuda_inc_path):
                    env['CUDA_INC_PATH'] = cuda_inc_path
                    print(f"[ComfyUI-HunyuanX] Set CUDA_INC_PATH={cuda_inc_path}")

            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", ".", "--no-build-isolation"],
                cwd=rasterizer_dir,
                capture_output=True,
                text=True,
                timeout=300,
                env=env
            )

            if result.returncode == 0:
                print("[ComfyUI-HunyuanX] ✅ custom_rasterizer compiled successfully")
            else:
                print("[ComfyUI-HunyuanX] ⚠️  custom_rasterizer compilation failed")
                print("[ComfyUI-HunyuanX] Texture generation may be limited")
                if result.stderr:
                    # Print more error details for debugging
                    print(f"[ComfyUI-HunyuanX] Error: {result.stderr[-500:]}")
        except Exception as e:
            print(f"[ComfyUI-HunyuanX] ⚠️  custom_rasterizer error: {e}")
    else:
        print("[ComfyUI-HunyuanX] ⚠️  custom_rasterizer directory not found")

    # 2. Compile mesh_inpaint_processor
    print("\n[ComfyUI-HunyuanX] Compiling mesh_inpaint_processor...")
    inpaint_dir = os.path.join(script_dir, "nodes", "lib", "hy3dpaint", "DifferentiableRenderer")

    if os.path.exists(os.path.join(inpaint_dir, "setup.py")):
        try:
            result = subprocess.run(
                [sys.executable, "setup.py", "build_ext", "--inplace"],
                cwd=inpaint_dir,
                capture_output=True,
                text=True,
                timeout=300
            )

            if result.returncode == 0:
                print("[ComfyUI-HunyuanX] ✅ mesh_inpaint_processor compiled successfully")
            else:
                print("[ComfyUI-HunyuanX] ⚠️  mesh_inpaint_processor compilation failed")
                print("[ComfyUI-HunyuanX] Texture inpainting may not work")
                if result.stderr:
                    print(f"[ComfyUI-HunyuanX] Error: {result.stderr[:300]}")
        except Exception as e:
            print(f"[ComfyUI-HunyuanX] ⚠️  mesh_inpaint_processor error: {e}")
    else:
        print("[ComfyUI-HunyuanX] ⚠️  mesh_inpaint_processor setup.py not found")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("🎨 ComfyUI-HunyuanX Installation")
    print("="*80 + "\n")

    # 1. Install requirements.txt first
    requirements_ok = install_requirements()

    # 2. Try to install flash_attn with smart wheel detection
    print("\n")
    flash_ok = try_install_flash_attn()

    # 3. Compile CUDA extensions
    print("\n")
    compile_cuda_extensions()

    # Print summary
    print("\n" + "="*80)
    print("📊 Installation Summary")
    print("="*80 + "\n")

    status_emoji = {True: "✅", False: "⚠️"}
    print(f"{status_emoji[requirements_ok]} Python dependencies (requirements.txt)")
    print(f"{status_emoji[flash_ok]} flash-attn (optional, 10-20% faster inference)")
    print("✅ CUDA extensions (check messages above for status)")

    if requirements_ok:
        print("\n" + "="*80)
        print("✅ ComfyUI-HunyuanX Installation Complete!")
        print("="*80 + "\n")
        print("Please restart ComfyUI to load the nodes.")
        print("\nNote: CUDA extensions may have failed - check messages above.")
        print("      Nodes will still work but some features may be limited.")
    else:
        print("\n" + "="*80)
        print("⚠️  Installation Completed with Errors")
        print("="*80 + "\n")
        print("Some dependencies failed to install.")
        print("Please check the error messages above and try manual installation.")

    print("\n")
