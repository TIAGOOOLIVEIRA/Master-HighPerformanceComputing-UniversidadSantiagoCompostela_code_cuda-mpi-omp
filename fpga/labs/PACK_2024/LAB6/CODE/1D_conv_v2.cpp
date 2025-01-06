#include "1D_conv.h"

void func(const dt  data_IN[N], dt data_OUT[N])
{
/* Conv weights */
    const float w1 = 0.2;
    const float w2 = 0.2;
    const float w3 = 0.2;
    const float w4 = 0.2;
    const float w5 = 0.2;

/* Initialization */
     dt d1 = data_IN[0];
     dt d2 = data_IN[1];
     dt d3 = data_IN[2];
     dt d4 = data_IN[3];


Loop: for (int i = 2; i < N - 2; i++) {
	#pragma HLS PIPELINE II=1

	dt d5 = data_IN[i + 2];
	dt RES = (w1*d1)+(w2*d2)+(w3*d3)+(w4*d4)+(w5*d5);
	d4=d5;
	d3=d4;
	d2=d3;
	d1=d2;
	data_OUT[i] = RES;
	}
}
