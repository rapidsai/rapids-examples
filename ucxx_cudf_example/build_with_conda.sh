#!/bin/bash
# =============================================================================
# build_with_conda.sh - Build UCXX cuDF Example using Conda
# =============================================================================
#
# This script sets up a conda environment with libcudf and libucxx installed
# from conda channels (no source compilation required), then builds the example.
#
# Prerequisites:
#   - Conda (Miniconda or Anaconda) installed
#   - NVIDIA GPU with CUDA support
#
# Usage:
#   ./build_with_conda.sh          # Build with default settings
#   ./build_with_conda.sh clean    # Clean build directory and rebuild
#
# =============================================================================

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="ucxx_cudf_example"
BUILD_DIR="${SCRIPT_DIR}/build"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== UCXX cuDF Example - Conda Build ===${NC}"

# -----------------------------------------------------------------------------
# Check prerequisites
# -----------------------------------------------------------------------------
if ! command -v conda &> /dev/null; then
    echo -e "${RED}Error: conda is not installed or not in PATH${NC}"
    echo "Please install Miniconda or Anaconda first."
    exit 1
fi

# Initialize conda for the current shell
eval "$(conda shell.bash hook)"

# -----------------------------------------------------------------------------
# Handle clean build
# -----------------------------------------------------------------------------
if [[ "$1" == "clean" ]]; then
    echo -e "${YELLOW}Cleaning build directory...${NC}"
    rm -rf "${BUILD_DIR}"
fi

# -----------------------------------------------------------------------------
# Create or update conda environment
# -----------------------------------------------------------------------------
if conda env list | grep -q "^${ENV_NAME} "; then
    echo -e "${YELLOW}Conda environment '${ENV_NAME}' already exists.${NC}"
    echo "Activating existing environment..."
else
    echo -e "${GREEN}Creating conda environment '${ENV_NAME}'...${NC}"
    echo "This will install libcudf, libucxx, rmm, and CUDA tools from conda."
    echo "This may take several minutes on first run..."
    conda env create -f "${SCRIPT_DIR}/conda/${ENV_NAME}.yml"
fi

# Activate the environment
echo -e "${GREEN}Activating conda environment...${NC}"
conda activate "${ENV_NAME}"

# -----------------------------------------------------------------------------
# Configure with CMake
# -----------------------------------------------------------------------------
echo -e "${GREEN}Configuring with CMake...${NC}"
cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES=native \
    -S "${SCRIPT_DIR}" \
    -B "${BUILD_DIR}"

# -----------------------------------------------------------------------------
# Build
# -----------------------------------------------------------------------------
echo -e "${GREEN}Building...${NC}"
cmake --build "${BUILD_DIR}" -j"$(nproc)"

# -----------------------------------------------------------------------------
# Done
# -----------------------------------------------------------------------------
echo ""
echo -e "${GREEN}=== Build Complete ===${NC}"
echo ""
echo "To run the example:"
echo "  conda activate ${ENV_NAME}"
echo "  ./build/ucxx_cudf_example"
echo ""
echo "Run with custom parameters:"
echo "  ./build/ucxx_cudf_example -s 10000    # 10000 elements"
echo "  ./build/ucxx_cudf_example -p 54321    # Use port 54321"
echo "  ./build/ucxx_cudf_example -h          # Show help"

