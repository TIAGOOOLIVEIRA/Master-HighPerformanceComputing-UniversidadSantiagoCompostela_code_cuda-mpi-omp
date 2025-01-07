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
```

### Memory of work
- Node reservation
  - compute --fpga_u250

- Set env variables
  - source /opt/cesga/Vitis/Vitis/2023.2/settings64.sh
  - export PLATFORM_REPO_PATHS=/opt/xilinx/platforms
  - source /opt/xilinx/xrt/setup.sh

- First execution without directives opmitization
  - v++ -c --mode hls --config hls_config.cfg --work_dir BASE
  - cat BASE/reports/hls_compile.rpt
  - Performance & Resource Estimates: 


| Modules & Loops                              | Issue Type | Slack  | Latency (cycles) | Latency (ns) | Iteration Latency | Interval | Trip Count | Pipelined | BRAM | DSP       | FF          | LUT         | URAM |
|----------------------------------------------|------------|--------|------------------|--------------|-------------------|----------|------------|-----------|------|-----------|-------------|-------------|------|
| + func                                       | Timing     | -0.12  | 48065            | 2.403e+05    | -                 | 48066    | -          | no        | -    | 10 (~0%)  | 2613 (~0%)  | 1937 (~0%)  | -    |
|   + func_Pipeline_VITIS_LOOP_6_1_PIPE_LOOP   | Timing     | -0.12  | 48059            | 2.403e+05    | -                 | 48059    | -          | no        | -    | 10 (~0%)  | 2093 (~0%)  | 1837 (~0%)  | -    |
|     o VITIS_LOOP_6_1_PIPE_LOOP               | II         | 4.25   | 48057            | 2.403e+05    | 43                | 5        | 9604       | yes       | -    | -         | -           | -           | -    |

  - HW Interfaces/AP_MEMORY: 

| Port              | Direction | Bitwidth |
|-------------------|-----------|----------|
| data_IN_address0  | out       | 14       |
| data_IN_address1  | out       | 14       |
| data_IN_q0        | in        | 32       |
| data_IN_q1        | in        | 32       |
| data_OUT_address0 | out       | 14       |
| data_OUT_d0       | out       | 32       |
| w_address0        | out       | 4        |
| w_address1        | out       | 4        |
| w_q0              | in        | 32       |
| w_q1              | in        | 32       |



- Second execution with directives opmitization for UNROLL in inner loops
  - v++ -c --mode hls --config hls_config.cfg --work_dir ARRAY_PARTITION
  - cat ARRAY_PARTITION/reports/hls_compile.rpt
  - Performance & Resource Estimates: 


# Performance & Resource Estimates

| Modules & Loops                             | Issue Type | Slack | Latency (cycles) | Latency (ns) | Iteration Latency | Interval | Trip Count | Pipelined | BRAM | DSP (~%) | FF (~%) | LUT (~%) | URAM |
|---------------------------------------------|------------|-------|------------------|--------------|-------------------|----------|------------|-----------|------|----------|---------|----------|------|
| + func                                      | -          | 0.17  | 25,597           | 128,000      | -                 | 25,598   | -          | no        | -    | 45 (~0%) | 6,057 (~0%) | 5,807 (~0%) | -    |
|   + func_Pipeline_VITIS_LOOP_11_1_VITIS_LOOP_12_2 | -          | 1.63  | 311              | 1,555        | -                 | 311      | -          | no        | -    | -        | 339 (~0%) | 357 (~0%)  | -    |
|     o VITIS_LOOP_11_1_VITIS_LOOP_12_2       | -          | 4.25  | 309              | 1,545        | 11                | 1        | 300        | yes       | -    | -        | -        | -        | -    |
|   o VITIS_LOOP_18_3                         | -          | 4.25  | 25,284           | 126,400      | 258               | -        | 98         | no        | -    | -        | -        | -        | -    |
|     + func_Pipeline_VITIS_LOOP_19_4         | -          | 0.17  | 151              | 755          | -                 | 151      | -          | no        | -    | 45 (~0%) | 5,022 (~0%) | 3,679 (~0%) | -    |
|       o VITIS_LOOP_19_4                     | -          | 4.25  | 149              | 745          | 53                | 1        | 98         | yes       | -    | -        | -        | -        | -    |
|     + func_Pipeline_VITIS_LOOP_36_7         | -          | 1.76  | 102              | 510          | -                 | 102      | -          | no        | -    | -        | 88 (~0%)  | 672 (~0%)  | -    |
|       o VITIS_LOOP_36_7                     | -          | 4.25  | 100              | 500          | 2                 | 1        | 100        | yes       | -    | -        | -        | -        | -    |


- Resources
  - AWS for FPGA: https://github.com/aws/aws-fpga