#include "2D_conv.h"

void func(const dt w[K], const dt  data_IN[N][N], dt data_OUT[N][N])
{
   
for (int i = 1; i < N - 1; ++i) // Ignore boundaries conditions
	{
	PIPE_LOOP: for (int j = 1; j < N - 1; ++j)
		{
#pragma HLS PIPELINE II=1
	    	 dt accum=0;
	  	for (int k = -1; k <= 1; ++k)
	  		for (int l = -1; l<= 1; ++l)	  	
	   		{
	    		 accum+= w[(k+1)*3+(l+1)]*data_IN[i+k][j+l];
	   		}

	   	 data_OUT[i][j] = accum;
	        }
	       }

}
