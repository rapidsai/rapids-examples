# UCXX cuDF Example

This example demonstrates how to create a cuDF integer column using `cudf::sequence` and transfer it between endpoints using the UCXX communication library.

## Overview

The example:
1. Creates a cuDF column with integer sequence `[0, 1, 2, ..., N-1]` using `cudf::sequence`
2. Sets up a UCXX listener and client endpoint (loopback connection)
3. Sends the column data from the listener endpoint to the client endpoint
4. Verifies the received data matches the original sequence
5. Reconstructs a new cuDF column from the received data

This demonstrates a pattern useful for distributed GPU computing where cuDF DataFrames/columns need to be transferred between nodes.

## Dependencies

All dependencies are installed via conda (no source compilation required):
- **libcudf** (>=25.08) - RAPIDS cuDF library (requires C++20)
- **libucxx** (>=0.40) - UCX C++ bindings
- **rmm** (>=25.08) - RAPIDS Memory Manager
- **cuda-nvcc** - NVIDIA CUDA compiler (>=12.0)

## Quick Start

### Option 1: Build with Conda (Local)

```bash
./build_with_conda.sh
```

Then run:
```bash
conda activate ucxx_cudf_example
./build/ucxx_cudf_example
```

### Option 2: Build with Docker

```bash
./build_with_docker.sh
```

Then run:
```bash
docker run --gpus all --rm -it ucxx_cudf_example ./build/ucxx_cudf_example
```

## Usage

```
Usage: ucxx_cudf_example [parameters]

Creates a cuDF integer sequence column and sends it between endpoints
using UCXX communication library.

Parameters:
  -p <port>    Port number to listen at (default: 12345)
  -s <size>    Number of elements in the sequence (default: 1000)
  -h           Print this help
```

## Testing

### Basic Test
Run with default parameters (1000 elements):
```bash
./build/ucxx_cudf_example
```

### Test with Larger Data
```bash
# 100,000 elements (~400KB)
./build/ucxx_cudf_example -s 100000

# 1 million elements (~4MB)
./build/ucxx_cudf_example -s 1000000
```

### Test with Docker
```bash
# Basic test
docker run --gpus all --rm -it ucxx_cudf_example ./build/ucxx_cudf_example

# Test with 100K elements
docker run --gpus all --rm -it ucxx_cudf_example ./build/ucxx_cudf_example -s 100000

# Interactive shell for debugging
docker run --gpus all --rm -it ucxx_cudf_example bash
```

### Expected Output

A successful run produces output like:
```
=== UCXX cuDF Example ===
Port: 12345
Sequence size: 1000

Created cuDF sequence column with 1000 elements [0, 1, 2, ..., 999]
Column data size: 4000 bytes
Sender Column preview: [0, 1, 2, 3, 4, ..., 995, 996, 997, 998, 999]

Setting up UCXX communication...
Waiting for connection...
Server received connection request from 127.0.0.1:xxxxx
Connection established!

Performing wireup exchange...
Wireup complete!

Sending cuDF column data...
Transfer complete!

Receiver Column preview: [0, 1, 2, 3, 4, ..., 995, 996, 997, 998, 999]

Verifying received data...
Verification PASSED! All 1000 elements match.

Creating new cuDF column from received data...
Created new cuDF column with 1000 elements

=== Example completed successfully ===
```

The key indicators of success:
- `Verification PASSED!` - Data was transferred correctly
- `Example completed successfully` - All steps completed without errors

## Key Concepts

### cudf::sequence

Creates a column filled with a sequence of values:
```cpp
cudf::numeric_scalar<int32_t> init_scalar(0, true, stream);  // Start at 0
cudf::numeric_scalar<int32_t> step_scalar(1, true, stream);  // Step by 1
auto column = cudf::sequence(size, init_scalar, step_scalar);
```

### UCXX Tag Send/Receive

Tag-based messaging for point-to-point communication:
```cpp
// Send data
endpoint->tagSend(data_ptr, size, ucxx::Tag{tag_value});

// Receive data
endpoint->tagRecv(buffer_ptr, size, ucxx::Tag{tag_value}, ucxx::TagMaskFull);
```

## Extending This Example

To send more complex cuDF data structures:

1. **DataFrames**: Serialize column-by-column, sending metadata first (column names, types, sizes)
2. **Strings columns**: Use `cudf::strings_column_view` to access the offsets and char data separately
3. **Nullable columns**: Also transfer the null bitmask

## Project Structure

```
ucxx_cudf_example/
├── build_with_conda.sh     # Build script using conda
├── build_with_docker.sh    # Build script using Docker
├── CMakeLists.txt          # CMake build configuration
├── Dockerfile              # Docker image definition
├── README.md               # This file
├── conda/
│   └── ucxx_cudf_example.yml  # Conda environment specification
└── src/
    └── ucxx_cudf_example.cpp  # Main example source code
```

## License

Apache-2.0
