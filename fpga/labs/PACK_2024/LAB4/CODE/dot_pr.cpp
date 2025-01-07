#include "dot_pr.h"

void func(const dt  A[M], const dt  B[M], dt *Y)
{
	dt accum=0;
	Loop: for(int i=0; i<M; i++)
		{
  //#pragma HLS PIPELINE ii=1
		accum+=A[i]*B[i];
		}
	*Y=accum;
}
