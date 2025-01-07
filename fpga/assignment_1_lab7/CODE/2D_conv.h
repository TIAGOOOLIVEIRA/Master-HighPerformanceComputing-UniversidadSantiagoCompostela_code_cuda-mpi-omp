#include <ap_fixed.h>

typedef ap_fixed<16, 8> fixed_t;
typedef float dt;
#define N 100
#define K 9

void func(const dt w[K], const dt  data_IN[N][N], dt data_OUT[N][N]);