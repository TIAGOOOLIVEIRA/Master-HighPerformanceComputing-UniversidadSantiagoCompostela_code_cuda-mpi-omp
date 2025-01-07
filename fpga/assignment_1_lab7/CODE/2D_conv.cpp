#include "2D_conv.h"

void func(const dt w[K], const dt data_IN[N][N], dt data_OUT[N][N]) {
    //On-chip buffer, to improve performance via local memory is used
    dt buffer[N][N];

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
            dt accum1 = w[0] * buffer[i - 1][j - 1] + w[1] * buffer[i - 1][j] + w[2] * buffer[i - 1][j + 1];
            dt accum2 = w[3] * buffer[i][j - 1] + w[4] * buffer[i][j] + w[5] * buffer[i][j + 1];
            dt accum3 = w[6] * buffer[i + 1][j - 1] + w[7] * buffer[i + 1][j] + w[8] * buffer[i + 1][j + 1];

            dt accum = accum1 + accum2 + accum3;

            data_OUT[i][j] = accum;
        }
    }
}