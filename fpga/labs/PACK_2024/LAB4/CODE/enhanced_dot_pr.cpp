// Code with product and accum in different loops

#include "dot_pr.h"
#define P 5   //higher than latency (cycles) of accum

void func(const dt  A[M], const dt  B[M], dt *Y)
{	
	dt res=0;
	dt accum[P];
		
	op: for(int i=0; i<M; i++)
		{
		#pragma HLS PIPELINE II=1 
		dt part = (i<P)?0 : accum[i%P];
		accum[i%P]=part+(A[i]*B[i]);		
		}
		
	accum: for(int i=0; i<P; i++)
		{
			res +=accum[i];
		}
	*Y=res;
}
