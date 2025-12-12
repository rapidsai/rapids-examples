#!/bin/bash
# =============================================================================
# build_with_docker.sh - Build UCXX cuDF Example using Docker
# =============================================================================
#
# This script builds a Docker image containing the UCXX cuDF example.
# The Docker image includes:
#   - CUDA runtime and development tools
#   - Conda environment with libcudf, libucxx, rmm (installed from conda)
#   - Pre-built example binary
#
# Prerequisites:
#   - Docker installed and running
#   - NVIDIA Container Toolkit (for GPU support)
#
# Usage:
#   ./build_with_docker.sh              # Build image with default name
#   ./build_with_docker.sh my_image     # Build image with custom name
#   ./build_with_docker.sh --no-cache   # Build without Docker cache
#
# =============================================================================

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_IMAGE_NAME="ucxx_cudf_example"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Parse arguments
IMAGE_NAME="${DEFAULT_IMAGE_NAME}"
DOCKER_ARGS=""

for arg in "$@"; do
    case $arg in
        --no-cache)
            DOCKER_ARGS="--no-cache"
            ;;
        -*)
            DOCKER_ARGS="${DOCKER_ARGS} ${arg}"
            ;;
        *)
            IMAGE_NAME="${arg}"
            ;;
    esac
done

echo -e "${GREEN}=== UCXX cuDF Example - Docker Build ===${NC}"

# -----------------------------------------------------------------------------
# Check prerequisites
# -----------------------------------------------------------------------------
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Error: docker is not installed or not in PATH${NC}"
    exit 1
fi

if ! docker info &> /dev/null; then
    echo -e "${RED}Error: Docker daemon is not running${NC}"
    exit 1
fi

# -----------------------------------------------------------------------------
# Build Docker image
# -----------------------------------------------------------------------------
echo -e "${GREEN}Building Docker image '${IMAGE_NAME}'...${NC}"
echo "This will:"
echo "  1. Set up a CUDA development environment"
echo "  2. Install Miniconda"
echo "  3. Create conda environment with libcudf, libucxx, rmm"
echo "  4. Build the example"
echo ""
echo "This may take 10-20 minutes on first build..."
echo ""

cd "${SCRIPT_DIR}"
docker build ${DOCKER_ARGS} -t "${IMAGE_NAME}" .

# -----------------------------------------------------------------------------
# Done
# -----------------------------------------------------------------------------
echo ""
echo -e "${GREEN}=== Docker Build Complete ===${NC}"
echo ""
echo "Image: ${IMAGE_NAME}"
echo ""
echo "To run the example:"
echo "  docker run --gpus all --rm -it ${IMAGE_NAME} ./build/ucxx_cudf_example"
echo ""
echo "Run with custom parameters:"
echo "  docker run --gpus all --rm -it ${IMAGE_NAME} ./build/ucxx_cudf_example -s 10000"
echo ""
echo "Interactive shell:"
echo "  docker run --gpus all --rm -it ${IMAGE_NAME} bash"

