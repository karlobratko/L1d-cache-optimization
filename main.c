#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <time.h>
#include <stdbool.h>
#include <immintrin.h>

typedef const char *cstr;

static_assert(sizeof(char) == 1, "sizeof(char) != 1");
static_assert(sizeof(int)  == 4, "sizeof(int) != 4");

typedef char          i8;
typedef unsigned char u8;
typedef int           i32;
typedef unsigned int  u32;

static_assert(sizeof(float)  == 4, "sizeof(float) != 4" );
static_assert(sizeof(double) == 8, "sizeof(double) != 8");

typedef float  f32;
typedef double f64;

#define SSE_ALIGNMENT 16

#ifndef CACHE_LINE_SIZE
#define CACHE_LINE_SIZE 64
#endif

#define N_F32_PER_CACHE_LINE (CACHE_LINE_SIZE / sizeof(f32))

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

void defer_fclose(void *ptr) {
    FILE **pptr = (FILE **)ptr;

    if (*pptr != NULL) {
        free(*pptr);
    }
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
    for (u32 m = 0; m < M; m += N_F32_PER_CACHE_LINE) {
        for (u32 p = 0; p < P; p += N_F32_PER_CACHE_LINE) {
            for (u32 n = 0; n < N; n += N_F32_PER_CACHE_LINE) {

                f32 * restrict c_ptr = C + m * P + p;
                const f32 * restrict a_ptr = A + m * N + n;
                for (u32 m2 = 0; m2 < N_F32_PER_CACHE_LINE && m + m2 < M; m2++) {

                    const f32 * restrict b_ptr = B + n * P + p;
                    for (u32 n2 = 0; n2 < N_F32_PER_CACHE_LINE && n + n2 < N; n2++) {

                        for (u32 p2 = 0; p2 < N_F32_PER_CACHE_LINE && p + p2 < P; p2++) {
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

void matrix_mul_transposed_cacheline(const f32 *A, const f32 *BT, f32 *C, u32 M, u32 N, u32 P) {
    for (u32 m = 0; m < M; m += N_F32_PER_CACHE_LINE) {
        for (u32 p = 0; p < P; p += N_F32_PER_CACHE_LINE) {
            for (u32 n = 0; n < N; n += N_F32_PER_CACHE_LINE) {

                f32 * restrict c_ptr = C + m * P + p;
                const f32 * restrict a_ptr = A + m * N + n;
                for (u32 m2 = 0; m2 < N_F32_PER_CACHE_LINE && m + m2 < M; m2++) {

                    const f32 * restrict b_ptr = BT + p * N + n;
                    for (u32 p2 = 0; p2 < N_F32_PER_CACHE_LINE && p + p2 < P; p2++) {

                        f32 sum = 0.f;
                        for (u32 n2 = 0; n2 < N_F32_PER_CACHE_LINE && n + n2 < N; n2++) {
                            sum += *(a_ptr + n2) * *(b_ptr + n2);
                        }
                        *(c_ptr + p2) = sum;

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
    for (u32 m = 0; m < M; m += N_F32_PER_CACHE_LINE) {
        for (u32 p = 0; p < P; p += N_F32_PER_CACHE_LINE) {
            for (u32 n = 0; n < N; n += N_F32_PER_CACHE_LINE) {

                f32 * restrict c_ptr = C + m * P + p;
                const f32 * restrict a_ptr = A + m * N + n;
                for (u32 m2 = 0; m2 < N_F32_PER_CACHE_LINE && m + m2 < M; m2++) {

                    const f32 * restrict b_ptr = B + n * P + p;
                    for (u32 n2 = 0; n2 < N_F32_PER_CACHE_LINE && n + n2 < N; n2++) {

                        const __m128 _a = _mm_set1_ps(*(a_ptr + n2));
                        u32 p2 = 0;
                        for (; p2 + 3 < N_F32_PER_CACHE_LINE && p + p2 + 3 < P; p2 += 4) {
                            const __m128 _b = _mm_loadu_ps(b_ptr + p2);
                            const __m128 _c = _mm_loadu_ps(c_ptr + p2);
                            _mm_storeu_ps(c_ptr + p2, _mm_add_ps(_c, _mm_mul_ps(_a, _b)));
                        }

                        for (; p2 < N_F32_PER_CACHE_LINE && p + p2 < P; p2++) {
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

void matrix_mul_transposed_sse(const f32 *A, const f32 *BT, f32 *C, u32 M, u32 N, u32 P) {
    for (u32 m = 0; m < M; m += N_F32_PER_CACHE_LINE) {
        for (u32 p = 0; p < P; p += N_F32_PER_CACHE_LINE) {
            for (u32 n = 0; n < N; n += N_F32_PER_CACHE_LINE) {

                f32 * restrict c_ptr = C + m * P + p;
                const f32 * restrict a_ptr = A + m * N + n;
                for (u32 m2 = 0; m2 < N_F32_PER_CACHE_LINE && m + m2 < M; m2++) {

                    const f32 * restrict b_ptr = BT + n * P + p;
                    for (u32 p2 = 0; p2 < N_F32_PER_CACHE_LINE && p + p2 < P; p2++) {

                        f32 sum = 0.f;
                        __m128 _sum = _mm_setzero_ps();

                        u32 n2 = 0;
                        for (; n2 + 3 < N_F32_PER_CACHE_LINE && n + n2 + 3 < N; n2 += 4) {
                            const __m128 _a = _mm_loadu_ps(a_ptr + n2);
                            const __m128 _b = _mm_loadu_ps(b_ptr + n2);

                            _sum = _mm_add_ps(_sum, _mm_mul_ps(_a, _b));
                        }

                        for (; n2 < N_F32_PER_CACHE_LINE && n + n2 < N; n2++) {
                            sum += *(a_ptr + n2) * *(b_ptr + n2);
                        }

                        _sum = _mm_hadd_ps(_sum, _sum);
                        _sum = _mm_hadd_ps(_sum, _sum);
                        *(c_ptr + p2) = sum + _mm_cvtss_f32(_sum);

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
    for (u32 m = 0; m < M; m += N_F32_PER_CACHE_LINE) {
        for (u32 p = 0; p < P; p += N_F32_PER_CACHE_LINE) {
            for (u32 n = 0; n < N; n += N_F32_PER_CACHE_LINE) {

                f32 * restrict c_ptr = C + m * P + p;
                const f32 * restrict a_ptr = A + m * N + n;
                for (u32 m2 = 0; m2 < N_F32_PER_CACHE_LINE && m + m2 < M; m2++) {

                    const f32 * restrict b_ptr = B + n * P + p;
                    for (u32 n2 = 0; n2 < N_F32_PER_CACHE_LINE && n + n2 < N; n2++) {

                        const __m256 _a = _mm256_set1_ps(*(a_ptr + n2));
                        u32 p2 = 0;
                        for (; p2 + 7 < N_F32_PER_CACHE_LINE && p + p2 + 7 < P; p2 += 8) {
                            const __m256 _b = _mm256_loadu_ps(b_ptr + p2);
                            const __m256 _c = _mm256_loadu_ps(c_ptr + p2);
                            _mm256_storeu_ps(c_ptr + p2, _mm256_add_ps(_c, _mm256_mul_ps(_a, _b)));
                        }

                        for (; p2 < N_F32_PER_CACHE_LINE && p + p2 < P; p2++) {
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

void matrix_mul_transposed_avx(const f32 *A, const f32 *BT, f32 *C, u32 M, u32 N, u32 P) {
    for (u32 m = 0; m < M; m += N_F32_PER_CACHE_LINE) {
        for (u32 p = 0; p < P; p += N_F32_PER_CACHE_LINE) {
            for (u32 n = 0; n < N; n += N_F32_PER_CACHE_LINE) {

                f32 * restrict c_ptr = C + m * P + p;
                const f32 * restrict a_ptr = A + m * N + n;
                for (u32 m2 = 0; m2 < N_F32_PER_CACHE_LINE && m + m2 < M; m2++) {

                    const f32 * restrict b_ptr = BT + n * P + p;
                    for (u32 p2 = 0; p2 < N_F32_PER_CACHE_LINE && p + p2 < P; p2++) {

                        f32 sum = 0.f;
                        __m256 _sum256 = _mm256_setzero_ps();

                        u32 n2 = 0;
                        for (; n2 + 7 < N_F32_PER_CACHE_LINE && n + n2 + 7 < N; n2 += 8) {
                            const __m256 _a = _mm256_loadu_ps(a_ptr + n2);
                            const __m256 _b = _mm256_loadu_ps(b_ptr + n2);

                            _sum256 = _mm256_add_ps(_sum256, _mm256_mul_ps(_a, _b));
                        }

                        for (; n2 < N_F32_PER_CACHE_LINE && n + n2 < N; n2++) {
                            sum += *(a_ptr + n2) * *(b_ptr + n2);
                        }

                        _sum256 = _mm256_hadd_ps(_sum256, _sum256);
                        _sum256 = _mm256_hadd_ps(_sum256, _sum256);

                        const __m128 low  = _mm256_extractf128_ps(_sum256, 0);
                        const __m128 high = _mm256_extractf128_ps(_sum256, 1);

                        *(c_ptr + p2) = sum + _mm_cvtss_f32(_mm_add_ps(low, high));

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

    return total_ns;
}

#define MAX_FILENAME_LENGTH 256

void print_usage(cstr target) {
    printf("Usage: %s [options]\n", target);
    printf("Options:\n");
    printf("  --output FILE | -o FILE     Set output filename (default: output.csv)\n");
}

void parse_args(const cstr *argv, i32 argc, cstr *filename) {
    const cstr target = argv[0];

    for (i32 i = 1; i < argc; i++) {
        if ((strcmp("-o", argv[i]) == 0 || strcmp("--output", argv[i]) == 0) && i + 1 < argc) {
            *filename = argv[i + 1];
            i++;
        } else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            print_usage(target);
            exit(EXIT_FAILURE);
        }
    }
}

struct impl_descriptor {
    matrix_mul_f func;
    cstr name;
    bool uses_transposed;
};

void benchmark_matrix_mul_impls(cstr filename) {
    FILE *file defer(defer_fclose) = NULL;

    if (filename != NULL) {
        file = fopen(filename, "w");
        assert(file != NULL && "Failed to open CSV file.");
    }

    if (file != NULL) {
        fprintf(file, "implementation,matrix_size,duration_ns\n");
    }

    const u32 sizes[] = {32, 64, 96, 128, 192, 256, 384, 512, 1024};
    const u32 num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    const struct impl_descriptor implementations[] = {
        {matrix_mul,                      "naive",                false},
        {matrix_mul_transposed,           "trans_naive",          true},
        {matrix_mul_cacheline,            "cacheline",            false},
        {matrix_mul_transposed_cacheline, "trans_cacheline",      true},
        {matrix_mul_sse,                  "sse",                  false},
        {matrix_mul_transposed_sse,       "trans_sse",            true},
        {matrix_mul_avx,                  "avx",                  false},
        {matrix_mul_transposed_avx,       "trans_avx",            true}
    };
    const u32 num_impls = sizeof(implementations) / sizeof(implementations[0]);

    for (u32 impl_i = 0; impl_i < num_impls; impl_i++) {
        const struct impl_descriptor implementation = implementations[impl_i];
        printf("Benchmarking %s implementation...\n\n", implementation.name);

        for (u32 size_i = 0; size_i < num_sizes; size_i++) {
            const u32 size = sizes[size_i];

            const u32 M = size, N = size, P = size;
            printf("  Matrix sizes: %ux%u\n", size, size);

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

            u32 iterations;
            if (size <= 128) {
                iterations = 100;
            } else if (size <= 256) {
                iterations = 50;
            } else {
                iterations = 25;
            }

            f64 total_ns;
            if (implementation.uses_transposed) {
                total_ns = benchmark_matrix_mul(A, BT, C, M, N, P, iterations, implementation.func);
            } else {
                total_ns = benchmark_matrix_mul(A, B,  C, M, N, P, iterations, implementation.func);
            }

            const f64 avg_duration_ns = total_ns        / iterations;
            const f64 avg_duration_ms = avg_duration_ns / NS_PER_MS;

            printf("  Average time: %fns (%fms)\n\n", total_ns, avg_duration_ms);

            if (file != NULL) {
                fprintf(file, "%s,%u,%f\n", implementation.name, size, avg_duration_ns);
                fflush(file);
            }
        }
    }
}

i32 main(i32 argc, const cstr *argv) {
    cstr filename = NULL;
    parse_args(argv, argc, &filename);

    srand(time(NULL));

    benchmark_matrix_mul_impls(filename);

    return EXIT_SUCCESS;
}
