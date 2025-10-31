from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys
import os
import pybind11
class BuildExt(build_ext):
    def build_extensions(self):
        if sys.platform == 'win32':
            # Windows-specific compiler flags
            for ext in self.extensions:
                ext.extra_compile_args = ['/O2', '/Wall']
        else:
            # Linux/Mac flags
            for ext in self.extensions:
                ext.extra_compile_args = ['-O3', '-Wall', '-fPIC']
        build_ext.build_extensions(self)

setup(
    name="mesh_inpaint_processor",
    ext_modules=[
        Extension(
            "mesh_inpaint_processor",
            ["mesh_inpaint_processor.cpp"],
            include_dirs=[
                pybind11.get_include(),
                pybind11.get_include(user=True)
            ],
            language='c++'
        ),
    ],
    cmdclass={'build_ext': BuildExt},
)