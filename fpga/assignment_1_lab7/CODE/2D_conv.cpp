#include "2D_conv.h"

void func(const fixed_t w[K], const fixed_t data_IN[N][N], fixed_t data_OUT[N][N]) {
    //On-chip buffer, to improve performance via local memory is used
    fixed_t buffer[N][N];

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            buffer[i][j] = data_IN[i][j];
        }
    }

    //2D convolution leveraging tree-height reduction
    for (int i = 1; i < N - 1; ++i) {
        PIPE_LOOP: for (int j = 1; j < N - 1; ++j) {
            #pragma HLS PIPELINE II=1
            //accumulation via tree-height reduction
            fixed_t accum1 = 0, accum2 = 0, accum3 = 0;

            // Row 1 (k = 0)
            for (int l = 0; l < 3; ++l) {
                accum1 += w[0 * 3 + l] * buffer[i - 1][j + l - 1];
            }
            // Row 2 (k = 1)
            for (int l = 0; l < 3; ++l) {
                accum2 += w[1 * 3 + l] * buffer[i][j + l - 1];
            }
            // Row 3 (k = 2)
            for (int l = 0; l < 3; ++l) {
                accum3 += w[2 * 3 + l] * buffer[i + 1][j + l - 1];
            }

            fixed_t accum = accum1 + accum2 + accum3;

            data_OUT[i][j] = accum;
        }
    }
}