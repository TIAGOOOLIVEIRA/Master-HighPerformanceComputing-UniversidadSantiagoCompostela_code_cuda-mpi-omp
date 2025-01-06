#include "1D_conv.h"

void func(const dt  data_IN[N], dt data_OUT[N])
{

 const float w1 = 0.2;
 const float w2 = 0.2;
 const float w3 = 0.2;
 const float w4 = 0.2;
 const float w5 = 0.2;

Loop: for (int i = 2; i < N - 2; i++) {
	#pragma HLS PIPELINE II=1

	const dt d1 = data_IN[i - 2];
	const dt d2 = data_IN[i - 1];
	const dt d3 = data_IN[i];
	const dt d4 = data_IN[i + 1];
	const dt d5 = data_IN[i + 2];

	const dt RES = (w1*d1)+(w2*d2)+(w3*d3)+(w4*d4)+(w5*d5);

	data_OUT[i] = RES;
	}
}
