#include "mulmat2D.h"
void func(const dt A[N][K], const dt B[K][M], dt C[N][M])
{
 row: for (int n = 0; n < N; ++n) {
 col:   for (int m = 0; m < M; ++m) {
      dt acc = 0;          
 loop_K:     for (int k = 0; k < K; ++k) {
        acc += A[n][k] * B[k][m];
      }
      C[n][m] = acc;
    }
  }
}
