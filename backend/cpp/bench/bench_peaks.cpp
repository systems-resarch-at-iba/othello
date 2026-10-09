/**
 * @file bench_peaks.cpp
 * @brief Measures the single-thread fused multiply-add throughput and memory read bandwidth of this
 * machine.
 *
 * Usage: bench_peaks
 *
 * Prints one JSON object. The fp32 throughput is the median of ten runs, and
 * the best run is reported separately. The read bandwidth is measured for
 * buffers that fit the cache levels and for buffers the size of the network
 * weights.
 */

#include <immintrin.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

    /**
     * @brief Clock used for all timings.
     */
    using Clock = std::chrono::steady_clock;

    /**
     * @brief Seconds elapsed since a time point.
     *
     * @param start Earlier time point.
     * @return The elapsed time in seconds.
     */
    double seconds_since(Clock::time_point start) {
        return std::chrono::duration<double>(Clock::now() - start).count();
    }

    /**
     * @brief Single-thread fp32 throughput in GFLOP/s, counting a fused multiply-add as 2 FLOP.
     *
     * @return The throughput of each of ten runs, sorted ascending.
     */
    std::vector<double> gflops_runs() {
        constexpr int kChains = 12;
        constexpr long kIters = 20'000'000;
        const __m256 b = _mm256_set1_ps(1.0000001f);
        const __m256 c = _mm256_set1_ps(1e-7f);
        std::vector<double> runs;
        for (int attempt = 0; attempt < 10; ++attempt) {
            __m256 acc[kChains];
            for (int k = 0; k < kChains; ++k) {
                acc[k] = _mm256_set1_ps(static_cast<float>(k));
            }
            const auto start = Clock::now();
            for (long i = 0; i < kIters; ++i) {
#pragma GCC unroll 12
                for (int k = 0; k < kChains; ++k) {
                    acc[k] = _mm256_fmadd_ps(acc[k], b, c);
                }
            }
            const double elapsed = seconds_since(start);
            float sink[8];
            __m256 total = acc[0];
            for (int k = 1; k < kChains; ++k) {
                total = _mm256_add_ps(total, acc[k]);
            }
            _mm256_storeu_ps(sink, total);
            volatile float keep = sink[0];
            (void)keep;
            const double flops = static_cast<double>(kIters) * kChains * 8 * 2;
            runs.push_back(flops / elapsed / 1e9);
        }
        std::sort(runs.begin(), runs.end());
        return runs;
    }

    /**
     * @brief Best single-thread read bandwidth over a buffer.
     *
     * @param bytes Size of the buffer.
     * @return The best of ten runs in GB/s.
     */
    double read_gbps(size_t bytes) {
        float *buffer = static_cast<float *>(std::aligned_alloc(64, bytes));
        const size_t count = bytes / sizeof(float);
        for (size_t i = 0; i < count; ++i) {
            buffer[i] = static_cast<float>(i % 97) * 0.01f;
        }
        const size_t passes = std::max<size_t>(3, (1ULL << 30) / bytes);
        double best = 0.0;
        for (int attempt = 0; attempt < 10; ++attempt) {
            __m256 acc[8];
            for (int k = 0; k < 8; ++k) {
                acc[k] = _mm256_setzero_ps();
            }
            const auto start = Clock::now();
            for (size_t p = 0; p < passes; ++p) {
                for (size_t i = 0; i + 64 <= count; i += 64) {
#pragma GCC unroll 8
                    for (int k = 0; k < 8; ++k) {
                        acc[k] = _mm256_add_ps(acc[k], _mm256_load_ps(buffer + i + k * 8));
                    }
                }
            }
            const double elapsed = seconds_since(start);
            float sink[8];
            __m256 total = acc[0];
            for (int k = 1; k < 8; ++k) {
                total = _mm256_add_ps(total, acc[k]);
            }
            _mm256_storeu_ps(sink, total);
            volatile float keep = sink[0];
            (void)keep;
            best = std::max(best, static_cast<double>(bytes) * passes / elapsed / 1e9);
        }
        std::free(buffer);
        return best;
    }

}  // namespace

/**
 * @brief Prints the machine's throughput and bandwidth as JSON.
 *
 * @return 0.
 */
int main() {
    constexpr size_t kMiB = 1024 * 1024;
    const size_t sizes[] = {256 * 1024, 2 * kMiB, 8 * kMiB, 16 * kMiB, 64 * kMiB, 512 * kMiB};
    const std::vector<double> runs = gflops_runs();
    std::printf("{\"peak_gflops_fp32\": %.1f, \"peak_gflops_fp32_best\": %.1f, \"read_gbps\": {",
                runs[runs.size() / 2], runs.back());
    bool first = true;
    for (const size_t bytes : sizes) {
        std::printf("%s\"%zu\": %.1f", first ? "" : ", ", bytes, read_gbps(bytes));
        first = false;
    }
    std::printf("}}\n");
    return 0;
}
