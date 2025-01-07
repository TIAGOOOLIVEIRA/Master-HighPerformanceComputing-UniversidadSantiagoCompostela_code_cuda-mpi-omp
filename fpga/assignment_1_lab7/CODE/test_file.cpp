#include <iostream>
#include <chrono>
#include "2D_conv.h"

//For benchmarking
void test_func(const dt w[K], const dt data_IN[N][N], dt data_OUT[N][N]) {
    for (int i = 1; i < N - 1; ++i) {
        PIPE_LOOP: for (int j = 1; j < N - 1; ++j) {
            #pragma HLS PIPELINE II=1
	    float accum = 0;
            for (int k = -1; k <= 1; ++k) {
                for (int l = -1; l <= 1; ++l) {
                    accum += w[(k + 1) * 3 + (l + 1)] * data_IN[i + k][j + l];
                }
            }
            data_OUT[i][j] = accum;
        }
    }
}

int main() {
    dt w[K], data_IN[N][N], sw_OUT[N][N], hw_OUT[N][N];

    srand(42);

    std::cout << "Initializing weights and input data:\n";
    for (int i = 0; i < K; ++i) {
        w[i] = rand() % 10;
    }

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            data_IN[i][j] = i + j;
        }
    }

    std::cout << "Running function (test_func):\n";
    auto start_sw = std::chrono::high_resolution_clock::now();
    test_func(w, data_IN, sw_OUT);
    auto end_sw = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_sw = end_sw - start_sw;
    std::cout << "Function completed in: " << elapsed_sw.count() << " seconds\n";

    std::cout << "Running device function (func):\n";
    auto start_hw = std::chrono::high_resolution_clock::now();
    func(w, data_IN, hw_OUT);
    auto end_hw = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_hw = end_hw - start_hw;
    std::cout << "Device function completed in: " << elapsed_hw.count() << " seconds\n";

    std::cout << "Validating results:\n";
    bool success = true;
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            if (sw_OUT[i][j] != hw_OUT[i][j]) {
                std::cout << "Mismatch at (" << i << ", " << j << "): "
                          << "test_func_OUT = " << sw_OUT[i][j] << ", func_OUT = " << hw_OUT[i][j] << "\n";
                success = false;
            }
        }
    }

    // Output overall test result
    if (success) {
        std::cout << "All tests passed successfully!\n";
    } else {
        std::cout << "Test failed! See above for details.\n";
    }

    // Report benchmark comparison
    std::cout << "\n<<------BENCHMARK RESULT------>>\n";
    std::cout << "Software function (test_func): " << elapsed_sw.count() << " seconds\n";
    std::cout << "Hardware function (func): " << elapsed_hw.count() << " seconds\n";
    std::cout << "Speedup (test_func / func): " << elapsed_sw.count() / elapsed_hw.count() << "x\n";

    return 0;
}