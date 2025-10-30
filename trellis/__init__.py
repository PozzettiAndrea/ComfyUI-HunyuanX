"""
TRELLIS 3D Generation Module

Wrapper for Microsoft TRELLIS (Structured 3D Latents for Scalable and Versatile 3D Generation)

Project: https://github.com/microsoft/TRELLIS
Paper: https://arxiv.org/abs/2412.01506
License: MIT License
"""

import sys
import os
from pathlib import Path

# Add TRELLIS to Python path
trellis_path = Path(__file__).parent / "TRELLIS"
if str(trellis_path) not in sys.path:
    sys.path.insert(0, str(trellis_path))

# Set environment variables for TRELLIS
os.environ.setdefault('SPCONV_ALGO', 'native')  # Faster for single runs

__version__ = "0.1.0"
__all__ = []
