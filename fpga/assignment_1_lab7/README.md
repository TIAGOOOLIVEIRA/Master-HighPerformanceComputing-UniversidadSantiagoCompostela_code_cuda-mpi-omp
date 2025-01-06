# Assignment 1. Provide a solution for Lab7 to achieve an initiation interval II=1 for PIPE_LOOP.

- Requirements
  - The solution must be faster than the default proposal. 
  - Neither additional pragmas nor changes in the configuration file are allowed.
  - Upload your code to the corresponding assignment section at Aula CESGA. 



### Lab 7: 2D Convolution (N=100, K=9)

  ```c
void func(const dt w[K], const dt data_IN[N][N], dt data_OUT[N][N])
{
    for (int i = 1; i < N - 1; ++i)
    {
        for (int j = 1; j < N - 1; ++j)
        {
            dt accum = 0;
            for (int k = -1; k <= 1; ++k)
                for (int l = -1; l <= 1; ++l)
                {
                    accum += w[(k + 1) * 3 + (l + 1)] * data_IN[i + k][j + l];
                }
            data_OUT[i][j] = accum;
        }
    }
    for (int i = 0; i < N; ++i)
    {
        data_OUT[i][0] = data_IN[i][0];
        data_OUT[i][N - 1] = data_IN[i][N - 1];
        data_OUT[0][i] = data_IN[0][i];
        data_OUT[N - 1][i] = data_IN[N - 1][i];
    }
}


- Resources
  - AWS for FPGA: https://github.com/aws/aws-fpga