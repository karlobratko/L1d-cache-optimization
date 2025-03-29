# L1 Data Cache-Optimized Matrix Multiplication

This project focuses on optimizing matrix multiplication by leveraging L1 data cache characteristics, demonstrating how different cache-aware strategies can significantly improve performance.

## Overview

Matrix multiplication is a memory-bound operation that suffers severely from cache misses when implemented naively. This project specifically targets L1 data cache optimization techniques to improve performance:

1. **Naive implementation**: Standard triply-nested loops with poor cache utilization
2. **Transposed implementation**: Improves cache locality by pre-transposing the second matrix for sequential memory access patterns
3. **Cache-line blocking**: Divides matrices into blocks precisely sized to match L1 data cache lines (typically 64 bytes = 16 floats)
4. **SSE with cache awareness**: Combines 128-bit SIMD instructions with cache-line blocking to maximize both L1 cache utilization and parallel computation
5. **AVX with cache awareness**: Extends to 256-bit SIMD instructions while maintaining optimal cache-line alignment and access patterns

Each implementation builds upon the previous one to demonstrate the cumulative effect of different cache optimization techniques.

## Features

- L1 data cache-focused optimizations
- Cache-line aligned memory allocation using `posix_memalign`
- Blocking algorithms precisely tuned to L1 cache line size
- Memory access pattern optimization to reduce cache misses
- SIMD acceleration (SSE and AVX) aligned with cache line boundaries
- Automatic detection of L1 data cache line size at compile time
- Comprehensive benchmarking to measure impact of each cache optimization
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

### Memory Layout & Cache Considerations

All matrices are stored in row-major order, which is critical for the cache optimization strategy. The implementation:

- Aligns all matrix memory to the exact L1 data cache line size
- Processes data in chunks matching the size of L1 cache lines (`N_F32_IN_CL` floating-point values)
- Carefully handles matrices whose dimensions are not multiples of the cache line size
- Uses blocking factors derived directly from the L1 cache line size rather than arbitrary values
- Organizes computation to minimize TLB misses alongside cache misses

### L1 Cache Optimizations

1. **Cache Line Alignment**: All matrices are aligned to L1 cache line boundaries (typically 64 bytes) to eliminate split cache line accesses
2. **Cache Line Blocking**: Matrices are processed in blocks precisely sized to match L1 cache lines (`N_F32_IN_CL` float elements)
3. **Sequential Access Patterns**: Matrix traversal order optimized to maximize sequential accesses and minimize cache misses
4. **Matrix Transposition**: Pre-transposed matrices improve spatial locality by converting column-wise accesses to row-wise
5. **Minimized Cache Pollution**: Implementation carefully manages which data remains in cache during computation
6. **Vector Length Alignment**: SIMD vector operations aligned with cache line boundaries for optimal performance
7. **Non-temporal Prefetching**: Strategic prefetching with careful timing to preload data into L1 cache before needed

### Benchmarking & Cache Analysis

The benchmarking system is designed to accurately measure L1 cache performance:

- Initial warm-up phase preloads caches to measure steady-state performance
- Result matrix reset between runs to avoid contaminating measurements
- High-precision monotonic clock timing for accurate microsecond measurements
- Reports in both nanoseconds and milliseconds to highlight magnitude of improvements
- Optional high-priority execution mode to minimize system interference
- CPU pinning to avoid cache coherency effects between cores
- Non-power-of-2 matrix dimensions (131×131×131) to stress edge cases in cache line utilization

For deeper cache analysis, the code can be combined with hardware performance counters:

```bash
# Analyze L1 data cache performance
perf stat -e cycles,instructions,cache-references,cache-misses,L1-dcache-loads,L1-dcache-load-misses ./mm

# Analyze TLB impact
perf stat -e cycles,instructions,L1-dcache-loads,L1-dcache-load-misses,dTLB-loads,dTLB-load-misses ./mm
```

## Code Structure

The implementation progresses through increasingly sophisticated L1 cache optimizations:

- `matrix_mul`: Naive matrix multiplication with poor cache utilization
- `matrix_mul_transposed`: Restructures memory access for better spatial locality
- `matrix_mul_cacheline`: Implements blocking precisely matched to L1 cache line size
- `matrix_mul_sse`: Combines cache-line blocking with 4-wide SIMD operations
- `matrix_mul_avx`: Extends to 8-wide SIMD while maintaining cache-line optimization

Key constants:
- `CACHE_LINE_SIZE`: Automatically detected size of L1 data cache line (typically 64 bytes)
- `N_F32_IN_CL`: Number of 32-bit floats that fit in one cache line (typically 16 floats)

The implementation is carefully structured to:
1. Minimize cache thrashing (repeated loading/unloading of same cache lines)
2. Maximize spatial locality (accessing memory in sequential patterns)
3. Optimize temporal locality (reusing cached values as much as possible)
4. Handle edge cases where matrix dimensions aren't multiples of cache lines

## Performance Expectations & L1 Cache Impact

The performance improvements demonstrate the critical impact of L1 data cache optimization:

1. **Naive implementation**: Establishes baseline with poor cache utilization
2. **Transposed implementation**: Reduces column-wise cache misses
3. **Cache-line implementation**: Dramatically improves cache hit rate by blocking to L1 cache line size
4. **SSE with cache awareness**: Combines cache benefits with SIMD parallelism
5. **AVX with cache awareness**: Extends SIMD width while maintaining cache efficiency

These improvements are most dramatic when matrix dimensions exceed L1 cache size but still fit in L2/L3 cache. The non-power-of-2 dimensions (131×131×131) specifically test how the implementation handles cache line boundary edge cases.

You can observe cache behavior by using performance monitoring tools like `perf`:

```bash
perf stat -e L1-dcache-loads,L1-dcache-load-misses ./mm
```

On modern CPUs with sufficient L1 cache, the optimized implementations should show L1 data cache miss rates below 5% compared to 70-80% for the naive implementation.
