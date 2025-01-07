#include "vec_sum.h"

void func(const dt  A[M], const dt  B[M], dt Y[M])
{
/* uncomment the HLS ARRAY_PARTITION pragmasfor full unrolling. Try unroll with and without these pragmas. You should observe that it is not possible a full unrolling because of memory constraints on read/write ports. */ 

/*
#pragma HLS ARRAY_PARTITION variable=Y complete dim=1
#pragma HLS ARRAY_PARTITION variable=B complete dim=1
#pragma HLS ARRAY_PARTITION variable=A complete dim=1
*/	

loop:for(int i=0; i<M; i++)
		{
/* Try first with pipeline and later with unroll. Compare results.*/
//#pragma HLS PIPELINE II=1  
//#pragma HLS UNROLL
		Y[i]=A[i]+B[i];
		}
}
