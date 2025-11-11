# HunyuanX Tests

Simple, clean test structure.

## Structure

```
tests/
├── torun/              # All test files and utilities
│   ├── conftest.py     # pytest configuration and fixtures
│   ├── pytest.ini      # pytest settings
│   ├── test_*.py       # Test files
│   ├── render_utils.py # Test helpers
│   └── testutils/      # Test utilities
└── test_outputs/       # All test outputs (gitignored)
    ├── .gitkeep
    └── [generated files and directories]
```

## Running Tests

From the `tests/torun/` directory:

```bash
# Run all tests
pytest

# Run specific test file
pytest test_basic_nodes.py

# Run with verbose output
pytest -v

# Skip slow tests
pytest -m "not slow"
```

## What Gets Generated

All test outputs (renders, meshes, logs, etc.) are saved to `test_outputs/` and **ignored by git**.

Each test run creates a timestamped directory like:
- `test_outputs/2025-11-11_16-30-45/`
- `test_outputs/latest/` (symlink to most recent run)

## Clean and Simple

- ✅ All test code in `torun/`
- ✅ All outputs in `test_outputs/` (gitignored)
- ✅ No clutter in the tests/ root
- ✅ Easy to find and run tests
