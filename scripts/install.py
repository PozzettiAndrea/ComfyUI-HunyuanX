"""
ComfyUI-MeshCraft Installation Script

This script runs when the node is installed via ComfyUI-Manager.
It compiles the CUDA extension needed for texture generation.
"""

import subprocess
import sys
import os

def main():
    """Install ComfyUI-MeshCraft with CUDA extension compilation."""

    print("\n" + "="*80)
    print("📦 Installing ComfyUI-MeshCraft...")
    print("="*80 + "\n")

    # Get MeshCraft root directory (go up from scripts/)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    current_dir = os.path.dirname(script_dir)  # Go up to MeshCraft root

    # Install Python dependencies from requirements.txt
    requirements_txt = os.path.join(current_dir, "requirements.txt")
    if os.path.exists(requirements_txt):
        print("📥 Installing Python dependencies...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", requirements_txt
        ])
        print("✅ Dependencies installed\n")

    # Compile CUDA extension
    rasterizer_dir = os.path.join(current_dir, "lib", "hy3dpaint", "custom_rasterizer")
    setup_py = os.path.join(rasterizer_dir, "setup.py")

    if os.path.exists(setup_py):
        print("🔧 Compiling CUDA rasterizer extension...")
        print("   This may take 1-2 minutes...\n")

        original_dir = os.getcwd()
        try:
            os.chdir(rasterizer_dir)
            subprocess.check_call([sys.executable, "setup.py", "install"])
            print("\n✅ CUDA extension compiled successfully!")
        except subprocess.CalledProcessError as e:
            print(f"\n❌ CUDA extension compilation failed: {e}")
            print("   Texture generation nodes may not work.")
        finally:
            os.chdir(original_dir)

    print("\n" + "="*80)
    print("✅ ComfyUI-MeshCraft installation complete!")
    print("   Restart ComfyUI to load the nodes.")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
