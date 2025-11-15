# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.
# Hunyuan 3D does not impose any additional limitations beyond what is outlined
# in the repsective licenses of these third-party components.
# Users must comply with all terms and conditions of original licenses of these third-party
# components and must ensure that the usage of the third party components adheres to
# all relevant laws and regulations.

# For avoidance of doubts, Hunyuan 3D means the large language models and
# their software and algorithms, including trained model weights, parameters (including
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code,
# fine-tuning enabling code and other elements of the foregoing made publicly available
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.

from setuptools import setup, find_packages
import torch
import os
import shutil
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension

# Force use of system g++ instead of conda's GCC 14+ which has compatibility issues
system_gxx = shutil.which('g++', path='/usr/bin:/bin')
if system_gxx and os.path.exists(system_gxx):
    print(f"[custom_rasterizer] Using system g++: {system_gxx}")
    os.environ['CXX'] = system_gxx
    os.environ['CC'] = system_gxx.replace('g++', 'gcc')
else:
    print("[custom_rasterizer] Warning: System g++ not found, using default compiler")

# Auto-detect PyTorch's C++11 ABI setting
def get_pytorch_abi():
    """
    Detect the _GLIBCXX_USE_CXX11_ABI flag that PyTorch was compiled with.
    Returns '0' for old ABI or '1' for C++11 ABI.
    """
    try:
        # Try to get ABI directly from PyTorch (available in newer versions)
        if hasattr(torch._C, '_GLIBCXX_USE_CXX11_ABI'):
            abi_flag = '1' if torch._C._GLIBCXX_USE_CXX11_ABI else '0'
            print(f"[custom_rasterizer] Detected PyTorch ABI from torch._C: {abi_flag}")
            return abi_flag
    except:
        pass

    # Fallback: check torch library symbols
    try:
        import subprocess
        import os

        # Find libtorch_cpu.so or libc10.so
        torch_lib_dir = os.path.join(os.path.dirname(torch.__file__), 'lib')
        libc10_path = os.path.join(torch_lib_dir, 'libc10.so')

        if os.path.exists(libc10_path):
            # Check for C++11 ABI string symbol
            result = subprocess.run(['nm', '-D', libc10_path],
                                    capture_output=True, text=True, timeout=10)
            # Look for the C++11 ABI version of the symbol
            if '__cxx11' in result.stdout:
                print(f"[custom_rasterizer] Detected C++11 ABI from libc10.so symbols")
                return '1'
            else:
                print(f"[custom_rasterizer] Detected old ABI from libc10.so symbols")
                return '0'
    except Exception as e:
        print(f"[custom_rasterizer] Warning: Could not detect ABI from symbols: {e}")

    # Default: Use C++11 ABI (modern PyTorch default)
    print(f"[custom_rasterizer] Using default C++11 ABI (modern PyTorch standard)")
    return '1'

# Detect and set ABI flag
abi_flag = get_pytorch_abi()
abi_define = f'-D_GLIBCXX_USE_CXX11_ABI={abi_flag}'
print(f"[custom_rasterizer] Compiling with: {abi_define}")

# build custom rasterizer
custom_rasterizer_module = CUDAExtension(
    "custom_rasterizer_kernel",
    [
        "lib/custom_rasterizer_kernel/rasterizer.cpp",
        "lib/custom_rasterizer_kernel/grid_neighbor.cpp",
        "lib/custom_rasterizer_kernel/rasterizer_gpu.cu",
    ],
    extra_compile_args={
        'cxx': [abi_define],
        'nvcc': [abi_define, '-ccbin', '/usr/bin/g++']
    }
)

setup(
    packages=find_packages(),
    version="0.1",
    name="custom_rasterizer",
    include_package_data=True,
    package_dir={"": "."},
    ext_modules=[
        custom_rasterizer_module,
    ],
    cmdclass={"build_ext": BuildExtension},
)
