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


def compile_cuda_extensions():
    """
    Compile Hunyuan-specific CUDA extensions:
    - custom_rasterizer (for texture generation)
    - mesh_inpaint_processor (for texture inpainting)
    """
    print("\n[ComfyUI-HunyuanX] Compiling CUDA extensions...")
    print("[ComfyUI-HunyuanX] This may take 2-5 minutes...\n")

    script_dir = os.path.dirname(os.path.abspath(__file__))

    # 1. Compile custom_rasterizer
    print("[ComfyUI-HunyuanX] Compiling custom_rasterizer...")
    rasterizer_dir = os.path.join(script_dir, "nodes", "lib", "hy3dpaint", "custom_rasterizer")

    if os.path.exists(rasterizer_dir):
        try:
            result = subprocess.run(
                [sys.executable, "setup.py", "install"],
                cwd=rasterizer_dir,
                capture_output=True,
                text=True,
                timeout=300
            )

            if result.returncode == 0:
                print("[ComfyUI-HunyuanX] ✅ custom_rasterizer compiled successfully")
            else:
                print("[ComfyUI-HunyuanX] ⚠️  custom_rasterizer compilation failed")
                print("[ComfyUI-HunyuanX] Texture generation may be limited")
                if result.stderr:
                    print(f"[ComfyUI-HunyuanX] Error: {result.stderr[:300]}")
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
