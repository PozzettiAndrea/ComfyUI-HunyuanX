# ComfyUI-MeshCraft Tests

This directory contains the test suite for ComfyUI-MeshCraft.

## Quick Start

### Running Tests Locally

```bash
# From the ComfyUI-MeshCraft directory
cd /workspace/ComfyUI/custom_nodes/ComfyUI-MeshCraft

# Install development dependencies
pip install -r requirements-dev.txt

# Run all tests
pytest tests/ -v

# Run with coverage report
pytest tests/ -v --cov=. --cov-report=term-missing

# Run only fast tests (skip slow model-loading tests)
pytest tests/ -v -m "not slow"

# Run specific test file
pytest tests/test_basic_nodes.py -v

# Run specific test
pytest tests/test_basic_nodes.py::TestMeshPostprocessing::test_mesh_decimation_reduces_face_count -v
```

## Test Structure

```
tests/
├── __init__.py           # Test package init
├── conftest.py           # Shared fixtures and test configuration
├── test_basic_nodes.py   # Basic node functionality tests
└── README.md             # This file
```

## Writing Tests

### Basic Test Pattern

```python
def test_your_feature(sample_trimesh):
    """Test description."""
    from nodes import YourNode

    node = YourNode()
    result = node.process(sample_trimesh)

    assert result is not None
    # Add more assertions
```

### Available Fixtures

See `conftest.py` for all fixtures. Common ones:

- `sample_trimesh` - A simple cube mesh for testing
- `sample_image_tensor` - A random 512x512 image in ComfyUI format
- `sample_pil_image` - A PIL Image for testing
- `temp_mesh_file` - A temporary mesh file path
- `sample_mesh_with_uvs` - A mesh with UV coordinates

### Testing Patterns

#### 1. Test Input Validation

```python
def test_negative_input_rejected(self, sample_trimesh):
    """Test that negative values are rejected."""
    from nodes import YourNode

    node = YourNode()

    with pytest.raises((ValueError, AssertionError)):
        node.process(sample_trimesh, -100)
```

#### 2. Test Basic Functionality

```python
def test_process_returns_expected_shape(self, sample_trimesh):
    """Test that processing returns expected output."""
    from nodes import YourNode

    node = YourNode()
    result = node.process(sample_trimesh)

    assert len(result) == 2  # Node returns (mesh, preview)
    assert result[0] is not None
```

#### 3. Mock Heavy Operations

```python
@pytest.mark.skip(reason="Requires model files")
def test_with_mocked_model(self):
    """Test with mocked heavy model loading."""
    from unittest.mock import patch, MagicMock

    with patch('your_module.load_model') as mock_load:
        mock_model = MagicMock()
        mock_load.return_value = mock_model

        # Your test code here
```

#### 4. Test Edge Cases

```python
def test_handles_empty_mesh(self):
    """Test that node handles edge cases gracefully."""
    import trimesh

    from nodes import YourNode

    # Create empty mesh
    empty_mesh = trimesh.Trimesh()

    node = YourNode()

    # Should either handle gracefully or raise clear error
    with pytest.raises(ValueError, match="empty"):
        node.process(empty_mesh)
```

## Test Markers

Use markers to categorize tests:

- `@pytest.mark.slow` - Tests that take a long time (model loading, etc.)
- `@pytest.mark.skip(reason="...")` - Tests to skip with reason
- `@pytest.mark.gpu` - Tests that require GPU (for future)

Run specific markers:
```bash
# Run only slow tests
pytest tests/ -v -m "slow"

# Skip slow tests
pytest tests/ -v -m "not slow"
```

## CI/CD Integration

Tests run automatically on GitHub Actions:

- **On every push** to main/master
- **On every pull request**
- **Matrix testing**: Ubuntu, Windows, macOS × Python 3.10, 3.11, 3.12

View results at: https://github.com/YOUR_USERNAME/ComfyUI-MeshCraft/actions

### CI Test Jobs

1. **test** - Runs pytest on all platforms/Python versions
2. **lint** - Runs Ruff linting
3. **test-import** - Verifies nodes can be imported

## Best Practices

### ✅ DO

- Test public API (node INPUT_TYPES, FUNCTION methods)
- Test edge cases (empty inputs, None values, boundary conditions)
- Use fixtures for common test data
- Mock heavy operations (model loading, GPU operations)
- Write descriptive test names and docstrings
- Keep tests fast (< 1s per test ideally)

### ❌ DON'T

- Test private implementation details
- Require actual model files in tests
- Require GPU for tests (use CPU only)
- Make tests dependent on external services
- Write tests that modify global state
- Commit large test files (use fixtures instead)

## Debugging Failed Tests

### Verbose output
```bash
pytest tests/ -vv -s
```

### Run specific test with full traceback
```bash
pytest tests/test_basic_nodes.py::test_name -vv --tb=long
```

### Drop into debugger on failure
```bash
pytest tests/ --pdb
```

### See print statements
```bash
pytest tests/ -s
```

## Coverage

Generate HTML coverage report:
```bash
pytest tests/ --cov=. --cov-report=html
open htmlcov/index.html  # View in browser
```

Aim for:
- **80%+ coverage** for critical paths
- **100% coverage** for data processing functions
- Don't obsess over 100% (diminishing returns)

## Contributing

When adding new nodes:

1. **Add tests** for the new node in `test_basic_nodes.py` or a new test file
2. **Run tests locally** before committing
3. **Ensure CI passes** before merging PRs

When modifying existing nodes:

1. **Update tests** if behavior changes
2. **Add regression tests** if fixing bugs
3. **Verify coverage** doesn't decrease

## Troubleshooting

### Import errors in tests

Make sure you're in the right directory:
```bash
cd /workspace/ComfyUI/custom_nodes/ComfyUI-MeshCraft
pytest tests/
```

### Missing dependencies

```bash
pip install -r requirements-dev.txt
```

### CUDA errors

Tests should run on CPU only (handled by `use_cpu_only` fixture). If you see CUDA errors, check that the fixture is working.

### Fixture not found

Make sure `conftest.py` is in the tests directory and check fixture names match.

## Resources

- [Pytest documentation](https://docs.pytest.org/)
- [Pytest fixtures guide](https://docs.pytest.org/en/stable/fixture.html)
- [ComfyUI test examples](../../tests-unit/)
- [unittest.mock documentation](https://docs.python.org/3/library/unittest.mock.html)
