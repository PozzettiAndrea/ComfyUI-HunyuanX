# Testing Setup for ComfyUI-MeshCraft

## ✅ What We've Created

1. **Test Infrastructure**:
   - `tests/` directory with fixtures and example tests
   - `tests/conftest.py` - Shared fixtures for mesh/image testing
   - `tests/test_fixtures.py` - Fixture validation tests
   - `tests/test_basic_nodes.py` - Example node tests (templates)

2. **CI/CD**:
   - `.github/workflows/test.yml` - GitHub Actions workflow
   - Runs on Ubuntu, Windows, macOS
   - Tests Python 3.10, 3.11, 3.12
   - **100% free** for public repos!

3. **Dependencies**:
   - `requirements-dev.txt` - Test dependencies
   - `pytest.ini` - Pytest configuration
   - Added `pyglet` to requirements.txt (fixes trimesh rendering)

## ⚠️ Current Issue

**Problem**: Pytest cannot run tests from inside a package that uses relative imports.

**Error**:
```
ImportError: attempted relative import with no known parent package
```

**Root Cause**: The `__init__.py` in ComfyUI-MeshCraft uses relative imports (`from .nodes import ...`), which pytest cannot handle when collecting tests from the `tests/` subdirectory.

## 🔧 Solutions

### Option 1: Run Tests from ComfyUI Root (Recommended)

Instead of running tests from inside ComfyUI-MeshCraft, run them from the ComfyUI root where the custom node is properly importable:

```bash
cd /workspace/ComfyUI

# Add the custom node parent to Python path
export PYTHONPATH=/workspace/ComfyUI/custom_nodes:$PYTHONPATH

# Run tests
pytest custom_nodes/ComfyUI-MeshCraft/tests/ -v
```

### Option 2: Move Tests Outside Package

Move `tests/` to `/workspace/ComfyUI-MeshCraft-tests/` (outside the package):

```bash
mv /workspace/ComfyUI/custom_nodes/ComfyUI-MeshCraft/tests /workspace/ComfyUI-MeshCraft-tests
cd /workspace/ComfyUI-MeshCraft-tests
pytest -v
```

Update `.github/workflows/test.yml` to checkout both repos.

### Option 3: Use sys.path Manipulation

Add this to `tests/conftest.py` at the very top:

```python
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
```

This is hacky but works for local testing.

### Option 4: Mock All Imports (Current Approach)

The `test_fixtures.py` file shows this approach - don't import actual nodes, just test with mocks. This works but limits what you can test.

## 📝 How to Proceed

### For Local Development:

**Quick Test** (tests fixtures only, no node imports):
```bash
cd /workspace/ComfyUI/custom_nodes/ComfyUI-MeshCraft
pytest tests/test_fixtures.py --ignore=__init__.py -v
```

**Full Test** (run from ComfyUI root):
```bash
cd /workspace/ComfyUI
PYTHONPATH=/workspace/ComfyUI:$PYTHONPATH pytest custom_nodes/ComfyUI-MeshCraft/tests/ -v
```

### For CI/CD:

The GitHub Actions workflow in `.github/workflows/test.yml` needs to be updated to run from the correct directory. Update the "Run tests" step:

```yaml
- name: Run tests
  working-directory: ${{ github.workspace }}/../..
  run: |
    export PYTHONPATH=$PWD:$PYTHONPATH
    pytest custom_nodes/ComfyUI-MeshCraft/tests/ -v
```

Or checkout ComfyUI in the workflow and install MeshCraft as a custom node.

## 🚀 Recommended Approach for ComfyUI Custom Nodes

After researching, **most ComfyUI custom nodes don't have automated tests**. If you want to add them:

1. **Keep it simple**: Test only critical functions, not every node
2. **Mock heavy operations**: Model loading, GPU operations
3. **Focus on logic**: Test data processing, coordinate transforms, edge cases
4. **Manual GUI testing**: Still primary method for custom nodes

### Minimal Viable Testing Setup:

```python
# tests/test_core_logic.py
import pytest
import numpy as np
import trimesh

def test_mesh_decimation_logic():
    """Test mesh decimation without loading nodes."""
    # Import the actual decimation function, not the node
    from nodes import some_helper_function

    mesh = trimesh.creation.box()
    result = some_helper_function(mesh, target_faces=6)
    assert len(result.faces) <= 6

# Run with: pytest tests/test_core_logic.py -v
```

## 📚 What You Learned

- ✅ How to structure tests for Python packages
- ✅ How to use pytest fixtures
- ✅ How to set up GitHub Actions CI
- ✅ How ComfyUI's architecture works (studied tests-unit/)
- ✅ Common pitfalls with relative imports in packages

## 🎯 Next Steps

1. **Decision**: Choose one of the 4 options above
2. **Simplify**: Start with just a few critical tests
3. **Iterate**: Add tests as you find bugs
4. **CI**: Get GitHub Actions working once tests pass locally

The infrastructure is ready - you just need to pick an approach for the import issue!

## 💡 Pro Tip

Consider extracting complex logic into separate utility modules (e.g., `utils/mesh_processing.py`) that don't depend on ComfyUI imports. These are much easier to test in isolation!

```python
# utils/mesh_ops.py
def decimate_mesh(mesh, target_faces):
    """Pure function - easy to test!"""
    ...
    return decimated_mesh

# nodes.py
from .utils.mesh_ops import decimate_mesh

class Hy3D21MeshlibDecimate:
    def process(self, mesh, target_faces):
        return (decimate_mesh(mesh, target_faces),)
```

Now you can test `utils/mesh_ops.py` without any ComfyUI dependencies!
