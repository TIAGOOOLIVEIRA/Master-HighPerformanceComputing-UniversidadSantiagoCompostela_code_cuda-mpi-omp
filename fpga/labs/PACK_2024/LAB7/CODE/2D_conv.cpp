#include "2D_conv.h"


void func(const dt w[K], const dt data_IN[N][N], dt data_OUT[N][N]) {
    dt buff[3][N];
#pragma HLS ARRAY_PARTITION variable=buff complete dim=1
#pragma HLS ARRAY_PARTITION variable=w complete dim=1
#pragma HLS ARRAY_PARTITION variable=data_IN complete dim=1
#pragma HLS ARRAY_PARTITION variadata_OUT complete dim=1
    
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < N; ++j) {
#pragma HLS PIPELINE II=1
            buff[i][j] = data_IN[i][j];
        }
    }

    for (int i = 1; i < N - 1; ++i) {
        for (int j = 1; j < N - 1; ++j) {
#pragma HLS PIPELINE II=1
            dt accum = 0;

            //convolution using the buff
            for (int k = 0; k < 3; ++k) {
                for (int l = 0; l < 3; ++l) {
#pragma HLS UNROLL
                    accum += w[k * 3 + l] * buff[k][j + l - 1];
                }
            }

            data_OUT[i][j] = accum;
        }

        //shift the buffer for the next row
        if (i < N - 2) {
            for (int j = 0; j < N; ++j) {
#pragma HLS PIPELINE II=1
                buff[0][j] = buff[1][j];
                buff[1][j] = buff[2][j];
                buff[2][j] = data_IN[i + 2][j];
            }
        }
    }
}

/*
void func(const dt w[K], const dt  data_IN[N][N], dt data_OUT[N][N])
{
#pragma HLS ARRAY_PARTITION variable=w complete dim=1
#pragma HLS ARRAY_PARTITION variable=data_IN complete dim=1  
#pragma HLS ARRAY_PARTITION variable=data_OUT complete dim=1  

for (int i = 1; i < N - 1; ++i) // Ignore boundaries conditions
	{
	PIPE_LOOP: for (int j = 1; j < N - 1; ++j)
		{
#pragma HLS PIPELINE II=1
	    	 dt accum=0;
	  	for (int k = -1; k <= 1; ++k)
	  		for (int l = -1; l<= 1; ++l)	  	
	   		{
			 //#pragma HLS UNROLL
	    		 accum+= w[(k+1)*3+(l+1)]*data_IN[i+k][j+l];
	   		}

	   	 data_OUT[i][j] = accum;
	        }
	       }

}
*/