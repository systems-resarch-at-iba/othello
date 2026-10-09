/**
 * @file bench_net.cpp
 * @brief Times the forward pass of one network variant and prints JSON.
 *
 * Usage: bench_net --weights FILE --boards FILE --variant reference|avx2 --iters N --reps R
 * [--batch B]
 *
 * The boards file holds float32 canonical boards, 64 values per board. Each
 * repetition runs `iters` evaluations, cycling through the boards, and
 * reports the mean time per evaluation in nanoseconds. With --batch the
 * vectorized pass evaluates B boards per call.
 */

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "othello/net.hpp"

namespace {

    /**
     * @brief Reads a file of float32 values.
     *
     * @param path File to read.
     * @return The values in the file.
     * @throws std::runtime_error When the file cannot be opened.
     */
    std::vector<float> read_floats(const std::string &path) {
        std::ifstream file(path, std::ios::binary);
        if (!file) {
            throw std::runtime_error("cannot open " + path);
        }
        std::vector<char> bytes((std::istreambuf_iterator<char>(file)),
                                std::istreambuf_iterator<char>());
        std::vector<float> values(bytes.size() / sizeof(float));
        std::memcpy(values.data(), bytes.data(), values.size() * sizeof(float));
        return values;
    }

    /**
     * @brief Value of a command-line option.
     *
     * @param argc Argument count.
     * @param argv Arguments.
     * @param name Option name, such as "--iters".
     * @return The argument after the option, or an empty string when absent.
     */
    std::string option(int argc, char **argv, const char *name) {
        for (int i = 1; i + 1 < argc; ++i) {
            if (std::strcmp(argv[i], name) == 0) {
                return argv[i + 1];
            }
        }
        return "";
    }

}  // namespace

/**
 * @brief Times the forward pass of one network variant and prints JSON.
 *
 * @param argc Argument count.
 * @param argv Arguments.
 * @return 0 on success, 2 for invalid arguments and 1 for an error.
 */
int main(int argc, char **argv) {
    try {
        const std::string weights = option(argc, argv, "--weights");
        const std::string boards_path = option(argc, argv, "--boards");
        const std::string variant = option(argc, argv, "--variant");
        const int iters = std::atoi(option(argc, argv, "--iters").c_str());
        const int reps = std::atoi(option(argc, argv, "--reps").c_str());
        if (weights.empty() || boards_path.empty() || iters <= 0 || reps <= 0 ||
            (variant != "reference" && variant != "avx2")) {
            std::fprintf(stderr, "usage: bench_net --weights F --boards F --variant "
                                 "reference|avx2 --iters N --reps R\n");
            return 2;
        }
        const bool use_reference = variant == "reference";
        if (!use_reference && !othello::avx2_supported()) {
            std::fprintf(stderr, "AVX2 is not supported on this CPU\n");
            return 2;
        }

        const std::vector<float> boards = read_floats(boards_path);
        const size_t board_count = boards.size() / othello::kSquares;
        if (board_count == 0) {
            std::fprintf(stderr, "no boards\n");
            return 2;
        }

        othello::Network net(weights);
        auto scratch = std::make_unique<othello::Scratch>();
        othello::Evaluation out;
        float sink = 0.0f;

        const int batch = std::max(1, std::atoi(option(argc, argv, "--batch").c_str()));
        if (batch > othello::kMaxBatch || (use_reference && batch != 1)) {
            std::fprintf(stderr, "bad --batch\n");
            return 2;
        }
        std::vector<float> gathered(static_cast<size_t>(batch) * othello::kSquares);
        othello::Evaluation outs[othello::kMaxBatch];

        const auto run = [&](int count) {
            for (int i = 0; i < count; i += batch) {
                for (int k = 0; k < batch; ++k) {
                    const float *board =
                        boards.data() + ((i + k) % board_count) * othello::kSquares;
                    std::memcpy(gathered.data() + k * othello::kSquares, board,
                                othello::kSquares * sizeof(float));
                }
                if (use_reference) {
                    net.forward_reference(*scratch, gathered.data(), out);
                    sink += out.value + out.logits[0];
                } else {
                    net.forward_batch(*scratch, gathered.data(), batch, outs);
                    sink += outs[0].value + outs[batch - 1].logits[0];
                }
            }
        };

        run(use_reference ? 5 : 300);
        std::printf("{\"variant\": \"%s\", \"batch\": %d, \"iters\": %d, \"ns_per_eval\": [",
                    variant.c_str(), batch, iters);
        for (int r = 0; r < reps; ++r) {
            const auto t0 = std::chrono::steady_clock::now();
            run(iters);
            const auto t1 = std::chrono::steady_clock::now();
            const double ns = std::chrono::duration<double, std::nano>(t1 - t0).count() / iters;
            std::printf("%s%.1f", r == 0 ? "" : ", ", ns);
        }
        std::printf("], \"sink\": %g}\n", static_cast<double>(sink));
        return 0;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
