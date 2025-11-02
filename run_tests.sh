#!/bin/bash
# GPU workflow testing for ComfyUI-MeshCraft
# Handles __init__.py conflict with pytest (import errors)
#
# Usage:
#   ./run_tests.sh test_workflows.py -v -m meshcraft
#   ./run_tests.sh test_workflows.py -v -k "hunyuan"

set -e

# Get script directory (repo root)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Safety check: Is __init__.py already backed up from previous failed run?
if [ -f "__init__.py.bak" ]; then
    echo "⚠️  WARNING: Found leftover __init__.py.bak from previous run"
    echo "   Restoring it now..."
    mv __init__.py.bak __init__.py
fi

# Safety check: Does __init__.py exist?
if [ ! -f "__init__.py" ]; then
    echo "❌ ERROR: __init__.py not found!"
    echo "   The custom node will not work without it."
    echo "   Please restore __init__.py before running tests."
    exit 1
fi

# Temporarily rename __init__.py to avoid pytest import conflicts
mv __init__.py __init__.py.bak
RENAMED=1

# Cleanup function with verification
cleanup() {
    if [ "$RENAMED" = "1" ]; then
        if [ -f "__init__.py.bak" ]; then
            mv __init__.py.bak __init__.py
            echo "✅ Restored __init__.py"
        else
            echo "❌ ERROR: __init__.py.bak not found during cleanup!"
            echo "   Custom node may not load in ComfyUI."
            echo "   Please check if __init__.py exists in the repo root."
        fi
    fi
}

# Always cleanup on exit (success, failure, or interrupt)
trap cleanup EXIT INT TERM

# Run pytest with all arguments passed through
cd tests
pytest "$@"
TEST_EXIT_CODE=$?

# Exit with pytest's exit code
exit $TEST_EXIT_CODE
