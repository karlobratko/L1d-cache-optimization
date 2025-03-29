# Matrix Multiplication Benchmarking

This project demonstrates and benchmarks various optimization techniques for matrix multiplication in C.

## Overview

Matrix multiplication is a fundamental operation in linear algebra and has numerous applications in scientific computing, computer graphics, machine learning, and more. This project implements several approaches to matrix multiplication with increasing levels of optimization:

1. **Naive implementation**: Standard triply-nested loops
2. **Transposed implementation**: Improves cache locality by pre-transposing the second matrix
3. **Cache-aware implementation**: Divides matrices into blocks that fit into cache lines
4. **SSE-optimized implementation**: Uses 128-bit SIMD instructions (SSE) to process 4 elements at once
5. **AVX-optimized implementation**: Uses 256-bit SIMD instructions (AVX) to process 8 elements at once

## Features

- Comprehensive benchmarking with warm-up phase
- Cache-line aligned memory allocation
- SIMD acceleration (SSE and AVX)
- Automatic cache line size detection
- GCC cleanup attribute for automatic resource management

## Build Requirements

- GCC compiler with GNU11 support
- POSIX-compliant system
- CPU with SSE and AVX instruction set support

## Building and Running

Build the project with:

```bash
make
```

Run the benchmark with:

```bash
make run
```

For optimal benchmarking performance (high priority, CPU pinning):

```bash
make run-priority
```

## Technical Details

### Memory Layout

Matrices are stored in row-major order. The implementation handles matrices whose dimensions are not multiples of the cache line size or SIMD vector width.

### Optimizations

1. **Memory Alignment**: All matrices are aligned to cache line boundaries for optimal memory access
2. **Cache Blocking**: The implementation processes blocks of matrices that fit in L1 cache
3. **SIMD Instructions**: Utilizes SSE and AVX vector processing for parallel computations
4. **Matrix Transposition**: Improves cache locality by aligning memory access patterns

### Benchmarking

The benchmarking function:
- Performs a warm-up phase to load data into caches
- Resets the result matrix between runs
- Uses high-precision monotonic clock for timing
- Reports results in both nanoseconds and milliseconds

## Code Structure

- `matrix_mul`: Naive matrix multiplication
- `matrix_mul_transposed`: Multiplication with pre-transposed second matrix
- `matrix_mul_cacheline`: Cache-aware blocking implementation
- `matrix_mul_sse`: SSE-optimized implementation (128-bit vectors)
- `matrix_mul_avx`: AVX-optimized implementation (256-bit vectors)

## Performance Expectations

Generally, you should observe performance improvements in this order (from slowest to fastest):

1. Naive implementation
2. Transposed implementation
3. Cache-line implementation
4. SSE implementation
5. AVX implementation

The relative performance gains will depend on your specific CPU architecture, cache sizes, and matrix dimensions.
