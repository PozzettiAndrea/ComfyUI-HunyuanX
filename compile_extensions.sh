#!/bin/bash

# ComfyUI-MeshCraft Extension Compiler
# Automatically compiles custom_rasterizer and DifferentiableRenderer modules
# Requires: nvcc (CUDA), g++ 12+, Python with PyTorch CUDA support

set -e  # Exit on error (disabled for individual checks)

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color
BOLD='\033[1m'

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERBOSE=false
FORCE=false
SKIP_VERIFY=false

# ============================================================================
# Helper Functions
# ============================================================================

print_header() {
    echo -e "${BOLD}${BLUE}=======================================${NC}"
    echo -e "${BOLD}${BLUE}  ComfyUI-MeshCraft Extension Compiler${NC}"
    echo -e "${BOLD}${BLUE}=======================================${NC}"
    echo ""
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

print_step() {
    echo -e "\n${BOLD}$1${NC}"
}

verbose_log() {
    if [ "$VERBOSE" = true ]; then
        echo -e "${BLUE}[VERBOSE]${NC} $1"
    fi
}

# ============================================================================
# Prerequisite Checking Functions
# ============================================================================

check_command() {
    local cmd=$1
    local name=$2

    if command -v "$cmd" &> /dev/null; then
        return 0
    else
        return 1
    fi
}

check_nvcc() {
    print_step "Checking NVIDIA CUDA Compiler (nvcc)..."

    if check_command "nvcc" "NVIDIA CUDA Compiler"; then
        local nvcc_version=$(nvcc --version | grep "release" | sed 's/.*release //' | sed 's/,.*//')
        print_success "nvcc found (CUDA $nvcc_version)"
        verbose_log "nvcc path: $(which nvcc)"
        return 0
    else
        print_error "nvcc not found"
        print_info "CUDA Toolkit is required to compile custom_rasterizer"
        print_info "Install from: https://developer.nvidia.com/cuda-downloads"
        return 1
    fi
}

check_gpp() {
    print_step "Checking G++ Compiler..."

    if check_command "g++" "G++ Compiler"; then
        local gpp_version=$(g++ --version | head -n1 | grep -oP '\d+\.\d+\.\d+' | head -n1)
        local major_version=$(echo "$gpp_version" | cut -d. -f1)

        print_success "g++ found (version $gpp_version)"
        verbose_log "g++ path: $(which g++)"

        if [ "$major_version" -lt 12 ]; then
            print_warning "g++ version is $gpp_version (recommended: 12+)"
            print_info "Compilation may still work, but version 12+ is recommended"
        fi
        return 0
    else
        print_error "g++ not found"
        print_info "Install g++ with: sudo apt-get install g++ (Ubuntu/Debian)"
        return 1
    fi
}

check_python() {
    print_step "Checking Python..."

    if check_command "python" "Python" || check_command "python3" "Python"; then
        local python_cmd="python"
        if ! command -v python &> /dev/null; then
            python_cmd="python3"
        fi

        local python_version=$($python_cmd --version 2>&1 | grep -oP '\d+\.\d+\.\d+')
        print_success "Python found (version $python_version)"
        verbose_log "Python path: $(which $python_cmd)"
        echo "$python_cmd" > /tmp/meshcraft_python_cmd
        return 0
    else
        print_error "Python not found"
        print_info "Install Python 3.10+ from: https://www.python.org/downloads/"
        return 1
    fi
}

check_pytorch_cuda() {
    print_step "Checking PyTorch with CUDA support..."

    local python_cmd=$(cat /tmp/meshcraft_python_cmd 2>/dev/null || echo "python")

    if $python_cmd -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
        local torch_version=$($python_cmd -c "import torch; print(torch.__version__)" 2>/dev/null)
        local cuda_version=$($python_cmd -c "import torch; print(torch.version.cuda)" 2>/dev/null)
        print_success "PyTorch found with CUDA support (PyTorch $torch_version, CUDA $cuda_version)"
        return 0
    else
        print_error "PyTorch with CUDA support not found"
        print_info "Install PyTorch with CUDA from: https://pytorch.org/get-started/locally/"
        return 1
    fi
}

check_pybind11() {
    print_step "Checking pybind11..."

    local python_cmd=$(cat /tmp/meshcraft_python_cmd 2>/dev/null || echo "python")

    if $python_cmd -c "import pybind11" 2>/dev/null; then
        local pybind_version=$($python_cmd -c "import pybind11; print(pybind11.__version__)" 2>/dev/null)
        print_success "pybind11 found (version $pybind_version)"
        return 0
    else
        print_error "pybind11 not found"
        print_info "Install with: pip install pybind11"
        return 1
    fi
}

check_blender() {
    print_step "Checking Blender (optional)..."

    if check_command "blender" "Blender"; then
        local blender_version=$(blender --version 2>&1 | head -n1 | grep -oP '\d+\.\d+\.\d+' | head -n1)
        print_success "Blender found (version $blender_version)"
        print_info "UV unwrapping features will be available"
        verbose_log "Blender path: $(which blender)"
        return 0
    else
        print_warning "Blender not found (optional)"
        print_info "UV unwrapping features require Blender: https://www.blender.org/download/"
        print_info "Extension compilation will still work without Blender"
        return 0  # Return 0 because it's optional
    fi
}

run_all_checks() {
    print_header

    local all_passed=true

    check_nvcc || all_passed=false
    check_gpp || all_passed=false
    check_python || all_passed=false
    check_pytorch_cuda || all_passed=false
    check_pybind11 || all_passed=false

    # Optional checks (don't affect all_passed)
    check_blender

    if [ "$all_passed" = false ]; then
        echo ""
        print_error "Some prerequisites are missing"
        if [ "$FORCE" = false ]; then
            print_info "Fix the issues above or use --force to attempt compilation anyway"
            exit 1
        else
            print_warning "Continuing anyway due to --force flag..."
        fi
    else
        echo ""
        print_success "All prerequisites satisfied!"
    fi
}

# ============================================================================
# Compilation Functions
# ============================================================================

compile_custom_rasterizer() {
    print_step "Compiling custom_rasterizer (CUDA extension)..."

    local rasterizer_dir="$SCRIPT_DIR/hy3dpaint/custom_rasterizer"

    if [ ! -d "$rasterizer_dir" ]; then
        print_error "Directory not found: $rasterizer_dir"
        return 1
    fi

    cd "$rasterizer_dir"
    verbose_log "Working directory: $(pwd)"

    local python_cmd=$(cat /tmp/meshcraft_python_cmd 2>/dev/null || echo "python")

    print_info "Running: $python_cmd setup.py install"

    if [ "$VERBOSE" = true ]; then
        $python_cmd setup.py install
    else
        $python_cmd setup.py install > /tmp/meshcraft_rasterizer_build.log 2>&1
    fi

    if [ $? -eq 0 ]; then
        print_success "custom_rasterizer compiled successfully"
        cd "$SCRIPT_DIR"
        return 0
    else
        print_error "custom_rasterizer compilation failed"
        print_info "Check log: /tmp/meshcraft_rasterizer_build.log"
        if [ "$VERBOSE" = false ]; then
            echo ""
            echo "Last 20 lines of error log:"
            tail -20 /tmp/meshcraft_rasterizer_build.log
        fi
        cd "$SCRIPT_DIR"
        return 1
    fi
}

compile_differentiable_renderer() {
    print_step "Compiling DifferentiableRenderer (C++ extension)..."

    local renderer_dir="$SCRIPT_DIR/hy3dpaint/DifferentiableRenderer"

    if [ ! -d "$renderer_dir" ]; then
        print_error "Directory not found: $renderer_dir"
        return 1
    fi

    cd "$renderer_dir"
    verbose_log "Working directory: $(pwd)"

    local python_cmd=$(cat /tmp/meshcraft_python_cmd 2>/dev/null || echo "python")

    print_info "Running: $python_cmd setup.py install"

    if [ "$VERBOSE" = true ]; then
        $python_cmd setup.py install
    else
        $python_cmd setup.py install > /tmp/meshcraft_renderer_build.log 2>&1
    fi

    if [ $? -eq 0 ]; then
        print_success "DifferentiableRenderer compiled successfully"
        cd "$SCRIPT_DIR"
        return 0
    else
        print_error "DifferentiableRenderer compilation failed"
        print_info "Check log: /tmp/meshcraft_renderer_build.log"
        if [ "$VERBOSE" = false ]; then
            echo ""
            echo "Last 20 lines of error log:"
            tail -20 /tmp/meshcraft_renderer_build.log
        fi
        cd "$SCRIPT_DIR"
        return 1
    fi
}

# ============================================================================
# Verification Functions
# ============================================================================

verify_installations() {
    if [ "$SKIP_VERIFY" = true ]; then
        print_warning "Skipping verification (--skip-verify)"
        return 0
    fi

    print_step "Verifying installations..."

    local python_cmd=$(cat /tmp/meshcraft_python_cmd 2>/dev/null || echo "python")
    local all_verified=true

    # Test custom_rasterizer_kernel
    print_info "Testing custom_rasterizer_kernel import..."
    if $python_cmd -c "import custom_rasterizer_kernel; print('Module loaded successfully')" 2>/dev/null; then
        print_success "custom_rasterizer_kernel verified"
    else
        print_error "custom_rasterizer_kernel import failed"
        all_verified=false
    fi

    # Test mesh_inpaint_processor
    print_info "Testing mesh_inpaint_processor import..."
    if $python_cmd -c "import mesh_inpaint_processor; print('Module loaded successfully')" 2>/dev/null; then
        print_success "mesh_inpaint_processor verified"
    else
        print_error "mesh_inpaint_processor import failed"
        all_verified=false
    fi

    echo ""
    if [ "$all_verified" = true ]; then
        print_success "All modules verified successfully!"
        return 0
    else
        print_error "Some modules failed verification"
        print_info "Try running with --verbose to see detailed compilation output"
        return 1
    fi
}

# ============================================================================
# Main Execution
# ============================================================================

show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -h, --help          Show this help message"
    echo "  -v, --verbose       Show detailed compilation output"
    echo "  -f, --force         Force compilation even if prerequisites are missing"
    echo "  --skip-verify       Skip import verification after compilation"
    echo ""
    echo "This script compiles the custom_rasterizer and DifferentiableRenderer"
    echo "extensions for ComfyUI-MeshCraft."
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_usage
            exit 0
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -f|--force)
            FORCE=true
            shift
            ;;
        --skip-verify)
            SKIP_VERIFY=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Main execution flow
main() {
    # Check prerequisites
    run_all_checks

    # Compile modules
    local compilation_failed=false

    if ! compile_custom_rasterizer; then
        compilation_failed=true
    fi

    if ! compile_differentiable_renderer; then
        compilation_failed=true
    fi

    # Verify installations
    if [ "$compilation_failed" = false ]; then
        echo ""
        verify_installations
        local verify_result=$?

        echo ""
        echo -e "${BOLD}${GREEN}============================================${NC}"
        if [ $verify_result -eq 0 ]; then
            echo -e "${BOLD}${GREEN}  Compilation completed successfully!${NC}"
        else
            echo -e "${BOLD}${YELLOW}  Compilation completed with warnings${NC}"
        fi
        echo -e "${BOLD}${GREEN}============================================${NC}"
        exit $verify_result
    else
        echo ""
        echo -e "${BOLD}${RED}============================================${NC}"
        echo -e "${BOLD}${RED}  Compilation failed${NC}"
        echo -e "${BOLD}${RED}============================================${NC}"
        print_info "Check the error messages above for details"
        exit 1
    fi
}

# Run main function
main
