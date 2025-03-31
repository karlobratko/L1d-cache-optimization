# L1 Data Cache-Optimized Matrix Multiplication

This project focuses on optimizing matrix multiplication by leveraging L1 data cache characteristics, demonstrating how different cache-aware strategies can significantly improve performance.

## Overview

Matrix multiplication is a memory-bound operation that suffers severely from cache misses when implemented naively. This project specifically targets L1 data cache optimization techniques to improve performance:

1. **Naive implementation**: Standard triply-nested loops with poor cache utilization
2. **Transposed implementation**: Improves cache locality by pre-transposing the second matrix for sequential memory access patterns
3. **Cache-line blocking**: Divides matrices into blocks precisely sized to match L1 data cache lines (typically 64 bytes = 16 floats)
4. **Cache-line with transposition**: Combines cache-line blocking with transposed matrix for optimal memory access patterns
5. **SSE with cache awareness**: Combines 128-bit SIMD instructions with cache-line blocking to maximize both L1 cache utilization and parallel computation
6. **SSE with transposition**: Leverages transposed matrices with SIMD for optimal cache coherence
7. **AVX with cache awareness**: Extends to 256-bit SIMD instructions while maintaining optimal cache-line alignment and access patterns
8. **AVX with transposition**: Combines AVX operations with transposed matrix for maximum performance

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

For optimal benchmarking performance (high priority, CPU pinning, requires sudo privilege):

```bash
make run-priority
```

## Technical Details

### Memory Layout & Cache Considerations

All matrices are stored in row-major order, which is critical for the cache optimization strategy. The implementation:

- Aligns all matrix memory to the exact L1 data cache line size
- Processes data in chunks matching the size of L1 cache lines (`N_F32_PER_CACHE_LINE` floating-point values)
- Carefully handles matrices whose dimensions are not multiples of the cache line size
- Uses blocking factors derived directly from the L1 cache line size rather than arbitrary values
- Organizes computation to minimize TLB misses alongside cache misses

### L1 Cache Optimizations

1. **Cache Line Alignment**: All matrices are aligned to L1 cache line boundaries (typically 64 bytes) to eliminate split cache line accesses
2. **Cache Line Blocking**: Matrices are processed in blocks precisely sized to match L1 cache lines (`N_F32_PER_CACHE_LINE` float elements)
3. **Sequential Access Patterns**: Matrix traversal order optimized to maximize sequential accesses and minimize cache misses
4. **Matrix Transposition**: Pre-transposed matrices improve spatial locality by converting column-wise accesses to row-wise
5. **Minimized Cache Pollution**: Implementation carefully manages which data remains in cache during computation
6. **Vector Length Alignment**: SIMD vector operations aligned with cache line boundaries for optimal performance
7. **Optimized Loop Orders**: In transposed implementations, the inner loop order is arranged to maximize sequential memory access

### Benchmarking & Cache Analysis

The benchmarking system is designed to accurately measure L1 cache performance:

- Initial warm-up phase preloads caches to measure steady-state performance
- Result matrix reset between runs to avoid contaminating measurements
- High-precision monotonic clock timing for accurate microsecond measurements
- Reports in both nanoseconds and milliseconds to highlight magnitude of improvements
- Optional high-priority execution mode to minimize system interference
- CPU pinning to avoid cache coherency effects between cores
- Non-power-of-2 matrix dimensions (257×257×257) to stress edge cases in cache line utilization

## Code Structure

The implementation progresses through increasingly sophisticated L1 cache optimizations:

- `matrix_mul`: Naive matrix multiplication with poor cache utilization
- `matrix_mul_transposed`: Restructures memory access for better spatial locality
- `matrix_mul_cacheline`: Implements blocking precisely matched to L1 cache line size
- `matrix_mul_transposed_cacheline`: Combines cache-line blocking with transposed B matrix for perfectly sequential access patterns
- `matrix_mul_sse`: Combines cache-line blocking with 4-wide SIMD operations
- `matrix_mul_transposed_sse`: Optimizes SIMD operations with transposed matrix and inner loop reordering
- `matrix_mul_avx`: Extends to 8-wide SIMD while maintaining cache-line optimization
- `matrix_mul_transposed_avx`: Maximizes performance with AVX instructions and transposed matrices

Key constants:
- `CACHE_LINE_SIZE`: Automatically detected size of L1 data cache line (typically 64 bytes)
- `N_F32_PER_CACHE_LINE`: Number of 32-bit floats that fit in one cache line (typically 16 floats)

The implementation is carefully structured to:
1. Minimize cache thrashing (repeated loading/unloading of same cache lines)
2. Maximize spatial locality (accessing memory in sequential patterns)
3. Optimize temporal locality (reusing cached values as much as possible)
4. Handle edge cases where matrix dimensions aren't multiples of cache lines

### Cache Issues with Non-Transposed Implementations

It's important to note that even with careful cache line blocking and prefetching, the non-transposed implementations still suffer from suboptimal cache behavior when accessing the B matrix and updating C matrix. This occurs because:

1. The access pattern for B is strided with jumps of P elements between consecutive memory accesses 
   - Hardware prefetchers often struggle to recognize this access pattern 
   - Software prefetching (`_mm_prefetch`) has limitations in fully addressing this issue 
2. C matrix is sequentially updated as B matrix elements are accessed

The transposed implementations directly solve this issue by reorganizing memory layout to ensure all memory accesses follow cache-friendly sequential patterns and that C matrix elements are calculated fully in every iteration reducing number of modifications per element. Additionally, the inner loop reordering for SIMD optimization in `matrix_mul_transposed_sse` and `matrix_mul_transposed_avx` maximizes vectorization efficiency by processing vectors of elements in the optimal dimension.

## Performance Expectations & L1 Cache Impact

The performance improvements demonstrate the critical impact of L1 data cache optimization. Here are the actual benchmarking results for matrices of size 257×257×257:

| Implementation                  | Execution Time (ms) | Speedup vs. Naive |
|---------------------------------|---------------------|-------------------|
| Naive implementation            | 9.089 ms            | 1.00×             |
| Transposed implementation       | 8.064 ms            | 1.13×             |
| Cache-line implementation       | 5.485 ms            | 1.66×             |
| Transposed cache-line           | 4.206 ms            | 2.16×             |
| SSE with cache awareness        | 2.243 ms            | 4.05×             |
| Transposed SSE                  | 1.660 ms            | 5.47×             |
| AVX with cache awareness        | 2.071 ms            | 4.39×             |
| Transposed AVX                  | 1.526 ms            | 5.95×             |

These results clearly demonstrate several key insights:

1. Simply transposing the B matrix provides a modest 13% performance improvement
2. Cache-line blocking delivers a significant 66% speedup over the naive implementation
3. Combining transposition with cache-line blocking yields a 2.16× speedup
4. SIMD vectorization with SSE provides an impressive 4× performance boost
5. When combining SIMD with transposition, we achieve 5.5-6× speedup over the baseline
6. The AVX implementation with transposition delivers the best performance, nearly 6× faster than the naive version

These improvements are most dramatic when matrix dimensions exceed L1 cache size but still fit in L2/L3 cache. The non-power-of-2 dimensions (257×257×257) specifically test how the implementation handles cache line boundary edge cases.

### Test System Specifications

The benchmarks were performed on system with following specifications:

- CPU: AMD Ryzen 7 PRO 7840U with Radeon 780M Graphics
- CPU Cores: 8 cores, 16 threads
- L1 Data Cache: 32 KiB per core (256 KiB total)
- L1 Instruction Cache: 32 KiB per core (256 KiB total)
- L2 Cache: 1 MiB per core (8 MiB total)
- L3 Cache: 16 MiB (shared)