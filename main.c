#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <time.h>
#include <stdbool.h>
#include <immintrin.h>

static_assert(sizeof(int) == 4, "sizeof(int) != 4");

typedef int          i32;
typedef unsigned int u32;

static_assert(sizeof(float)  == 4, "sizeof(float) != 4" );
static_assert(sizeof(double) == 8, "sizeof(double) != 8");

typedef float  f32;
typedef double f64;

#define SSE_ALIGNMENT 16

#ifndef CACHE_LINE_SIZE
#define CACHE_LINE_SIZE 64
#endif

#define N_F32_IN_CL (CACHE_LINE_SIZE / sizeof(f32))

#define MS_PER_S   1000
#define US_PER_S  (1000 * MS_PER_S)
#define NS_PER_S  (1000 * US_PER_S)
#define NS_PER_US  1000
#define NS_PER_MS (1000 * NS_PER_US)

#define defer(f) __attribute__ ((cleanup(f)))

void defer_free(void *ptr) {
    void **pptr = (void **)ptr;
    free(*pptr);
}

i32 rand_between(i32 min, i32 max){
   return min + rand() / (RAND_MAX / (max - min + 1) + 1);
}

typedef void (*matrix_mul_f)(const f32 *A, const f32 *B, f32 *C, u32 M, u32 N, u32 P);

void matrix_setzero(f32 *A, u32 M, u32 N) {
    memset(A, 0, M * N * sizeof(f32));
}

void matrix_setrand(f32 *A, u32 M, u32 N) {
    for (u32 m = 0; m < M; m++) {
        for (u32 n = 0; n < N; n++) {
            *(A + m * N + n) = (f32)rand_between(1, 10);
        }
    }
}

bool matrix_eq(const f32 *A, const f32 *B, u32 M, u32 N) {
    for (u32 m = 0; m < M; m++) {
        for (u32 n = 0; n < N; n++) {
            if (*(A + m * N + n) != *(B + m * N + n)) {
                return false;
            }
        }
    }
    return true;
}

void matrix_transpose(const f32 *A, f32 *B, u32 M, u32 N) {
    for (u32 m = 0; m < M; m++) {
        for (u32 n = 0; n < N; n++) {
            *(B + n * M + m) = *(A + m * N + n);
        }
    }
}

void matrix_print(const f32 *A, u32 M, u32 N) {
    for (u32 m = 0; m < M; m++) {
        for (u32 n = 0; n < N; n++) {
            printf("%.2f ", *(A + m * N + n));
        }
        putchar('\n');
    }
}

void matrix_mul(const f32 *A, const f32 *B, f32 *C, u32 M, u32 N, u32 P) {
    for (u32 m = 0; m < M; m++) {
        for (u32 p = 0; p < P; p++) {
            f32 sum = 0;
            for (u32 n = 0; n < N; n++) {
                sum += *(A + m * N + n) * *(B + n * P + p);
            }
            *(C + m * P + p) = sum;
        }
    }
}

void matrix_mul_transposed(const f32 *A, const f32 *BT, f32 *C, u32 M, u32 N, u32 P) {
    for (u32 m = 0; m < M; m++) {
        for (u32 p = 0; p < P; p++) {
            f32 sum = 0;
            for (u32 n = 0; n < N; n++) {
                sum += *(A + m * N + n) * *(BT + p * N + n);
            }
            *(C + m * P + p) = sum;
        }
    }
}

void matrix_mul_cacheline(const f32 *A, const f32 *B, f32 *C, u32 M, u32 N, u32 P) {
    for (u32 m = 0; m < M; m += N_F32_IN_CL) {
        for (u32 p = 0; p < P; p += N_F32_IN_CL) {
            for (u32 n = 0; n < N; n += N_F32_IN_CL) {

                f32 * restrict c_ptr = C + m * P + p;
                const f32 * restrict a_ptr = A + m * N + n;
                for (u32 m2 = 0; m2 < N_F32_IN_CL && m + m2 < M; m2++) {

                    const f32 * restrict b_ptr = B + n * P + p;
                    for (u32 n2 = 0; n2 < N_F32_IN_CL && n + n2 < N; n2++) {

                        for (u32 p2 = 0; p2 < N_F32_IN_CL && p + p2 < P; p2++) {
                            *(c_ptr + p2) += *(a_ptr + n2) * *(b_ptr + p2);
                        }

                        b_ptr += P;
                    }

                    a_ptr += N;
                    c_ptr += P;
                }
            }
        }
    }
}

void matrix_mul_sse(const f32 *A, const f32 *B, f32 *C, u32 M, u32 N, u32 P) {
    for (u32 m = 0; m < M; m += N_F32_IN_CL) {
        for (u32 p = 0; p < P; p += N_F32_IN_CL) {
            for (u32 n = 0; n < N; n += N_F32_IN_CL) {

                f32 * restrict c_ptr = C + m * P + p;
                const f32 * restrict a_ptr = A + m * N + n;
                for (u32 m2 = 0; m2 < N_F32_IN_CL && m + m2 < M; m2++) {
                    //_mm_prefetch(a_ptr + N, _MM_HINT_NTA);

                    const f32 * restrict b_ptr = B + n * P + p;
                    for (u32 n2 = 0; n2 < N_F32_IN_CL && n + n2 < N; n2++) {

                        const __m128 _a = _mm_set1_ps(*(a_ptr + n2));
                        u32 p2 = 0;
                        for (; p2 + 3 < N_F32_IN_CL && p + p2 + 3 < P; p2 += 4) {
                            const __m128 _b = _mm_loadu_ps(b_ptr + p2);
                            const __m128 _c = _mm_loadu_ps(c_ptr + p2);
                            _mm_storeu_ps(c_ptr + p2, _mm_add_ps(_c, _mm_mul_ps(_a, _b)));
                        }

                        for (; p2 < N_F32_IN_CL && p + p2 < P; p2++) {
                            *(c_ptr + p2) += *(a_ptr + n2) * *(b_ptr + p2);
                        }

                        b_ptr += P;
                    }

                    a_ptr += N;
                    c_ptr += P;
                }
            }
        }
    }
}

void matrix_mul_avx(const f32 *A, const f32 *B, f32 *C, u32 M, u32 N, u32 P) {
    for (u32 m = 0; m < M; m += N_F32_IN_CL) {
        for (u32 p = 0; p < P; p += N_F32_IN_CL) {
            for (u32 n = 0; n < N; n += N_F32_IN_CL) {

                f32 * restrict c_ptr = C + m * P + p;
                const f32 * restrict a_ptr = A + m * N + n;
                for (u32 m2 = 0; m2 < N_F32_IN_CL && m + m2 < M; m2++) {
                    //_mm_prefetch(a_ptr + N, _MM_HINT_NTA);

                    const f32 * restrict b_ptr = B + n * P + p;
                    for (u32 n2 = 0; n2 < N_F32_IN_CL && n + n2 < N; n2++) {

                        const __m256 _a = _mm256_set1_ps(*(a_ptr + n2));
                        u32 p2 = 0;
                        for (; p2 + 7 < N_F32_IN_CL && p + p2 + 7 < P; p2 += 8) {
                            const __m256 _b = _mm256_loadu_ps(b_ptr + p2);
                            const __m256 _c = _mm256_loadu_ps(c_ptr + p2);
                            _mm256_storeu_ps(c_ptr + p2, _mm256_add_ps(_c, _mm256_mul_ps(_a, _b)));
                        }

                        for (; p2 < N_F32_IN_CL && p + p2 < P; p2++) {
                            *(c_ptr + p2) += *(a_ptr + n2) * *(b_ptr + p2);
                        }

                        b_ptr += P;
                    }

                    a_ptr += N;
                    c_ptr += P;
                }
            }
        }
    }
}

f64 benchmark_matrix_mul(const f32 *A, const f32 *B, f32 *C, u32 M, u32 N, u32 P, u32 iterations, matrix_mul_f mul) {
    struct timespec start, end;
    f64 total_ns = 0.;

    // warmup
    for (u32 i = 0; i < 2; i++) {
        mul(A, B, C, M, N, P);
    }

    for (u32 i = 0; i < iterations; i++) {
        matrix_setzero(C, M, P);

        clock_gettime(CLOCK_MONOTONIC, &start);
        mul(A, B, C, M, N, P);
        clock_gettime(CLOCK_MONOTONIC, &end  );

        f64 elapsed_ns  = (end.tv_sec  - start.tv_sec ) * NS_PER_S;
            elapsed_ns += (end.tv_nsec - start.tv_nsec);

        total_ns += elapsed_ns;
    }

    return total_ns / iterations;
}

int main(void) {
    srand(time(NULL));

    const u32 iterations = 1000;
    const u32 M = 131, N = 131, P = 131;

    printf("M = %u, N = %u, P = %u\n", M, N, P);
    printf("Iterations: %u\n", iterations);

    f32 *A  defer(defer_free),
        *B  defer(defer_free),
        *BT defer(defer_free),
        *C  defer(defer_free);

    assert(posix_memalign((void **)&A,  CACHE_LINE_SIZE, M * N * sizeof(f32)) == 0);
    matrix_setrand(A, M, N);

    assert(posix_memalign((void **)&B,  CACHE_LINE_SIZE, N * P * sizeof(f32)) == 0);
    matrix_setrand(B, N, P);

    assert(posix_memalign((void **)&BT, CACHE_LINE_SIZE, N * P * sizeof(f32)) == 0);
    matrix_transpose(B, BT, N, P);

    assert(posix_memalign((void **)&C,  CACHE_LINE_SIZE, M * P * sizeof(f32)) == 0);
    matrix_setzero(C, M, P);

    {
        const f64 total_ns = benchmark_matrix_mul(A, B, C, M, N, P, iterations, matrix_mul);
        printf("Average multiplication time (original):       %fns (%fms)\n", total_ns, total_ns / NS_PER_MS);
    }

    {
        const f64 total_ns = benchmark_matrix_mul(A, BT, C, M, N, P, iterations, matrix_mul_transposed);
        printf("Average multiplication time (pre-transposed): %fns (%fms)\n", total_ns, total_ns / NS_PER_MS);
    }

    {
        const f64 total_ns = benchmark_matrix_mul(A, B, C, M, N, P, iterations, matrix_mul_cacheline);
        printf("Average multiplication time (sub-matrix):     %fns (%fms)\n", total_ns, total_ns / NS_PER_MS);
    }

    {
        const f64 total_ns = benchmark_matrix_mul(A, B, C, M, N, P, iterations, matrix_mul_sse);
        printf("Average multiplication time (sse):            %fns (%fms)\n", total_ns, total_ns / NS_PER_MS);
    }

    {
        const f64 total_ns = benchmark_matrix_mul(A, B, C, M, N, P, iterations, matrix_mul_avx);
        printf("Average multiplication time (avx):            %fns (%fms)\n", total_ns, total_ns / NS_PER_MS);
    }

    return EXIT_SUCCESS;
}
