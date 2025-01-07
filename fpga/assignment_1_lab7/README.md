# Assignment 1. Provide a solution for Lab7 to achieve an initiation interval II=1 for PIPE_LOOP.

- Requirements
  - The solution must be faster than the default proposal. 
  - Neither additional pragmas nor changes in the configuration file are allowed.
  - Upload your code to the corresponding assignment section at Aula CESGA. 



### Lab 7: 2D Convolution (N=100, K=9)

  ```c
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
  - cat BASE/hls/syn/report/func_csynth.rpt
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



- Second execution with opmitizations
  - v++ -c --mode hls --config hls_config.cfg --work_dir OPTIMIZED
  - cat OPTIMIZED/hls/syn/report/func_csynth.rpt
  - Performance & Resource Estimates: 



| Modules & Loops                              | Issue Type | Slack | Latency (cycles) | Latency (ns) | Iteration Latency | Interval | Trip Count | Pipelined | BRAM       | DSP       | FF          | LUT         | URAM |
|----------------------------------------------|------------|-------|------------------|--------------|-------------------|----------|------------|-----------|------------|-----------|-------------|-------------|------|
| + func                                       | -          | 0.12  | 29844            | 1.492e+05    | -                 | 29845    | -          | no        | 36 (~0%)   | 44 (~0%)  | 5469 (~0%)  | 5071 (~0%)  | -    |
|   + func_Pipeline_VITIS_LOOP_16_3_PIPE_LOOP  | -          | 0.12  | 9637             | 4.818e+04    | -                 | 9637     | -          | no        | -          | 44 (~0%)  | 4793 (~0%)  | 4392 (~0%)  | -    |
|     o VITIS_LOOP_16_3_PIPE_LOOP              | -          | 4.25  | 9635             | 4.818e+04    | 33                | 1        | 9604       | yes       | -          | -         | -           | -           | -    |
| o VITIS_LOOP_8_1                             | -          | 4.25  | 20200            | 1.010e+05    | 202               | -        | 100        | no        | -          | -         | -           | -           | -    |
|   o VITIS_LOOP_9_2                           | -          | 4.25  | 200              | 1.000e+03    | 2                 | -        | 100        | no        | -          | -         | -           | -           | -    |

  - HW Interfaces/AP_MEMORY: 

| Port              | Direction | Bitwidth |
|-------------------|-----------|----------|
| data_IN_address0  | out       | 14       |
| data_IN_q0        | in        | 32       |
| data_OUT_address0 | out       | 14       |
| data_OUT_d0       | out       | 32       |
| w_address0        | out       | 4        |
| w_address1        | out       | 4        |
| w_q0              | in        | 32       |
| w_q1              | in        | 32       |

----------------------------------------------------
### Analysis 
Given the defined constraints for optimizing the proposed kernel without additional pragma directive, I decided to hiphotetize that some improvement in the performance could be achieved by leveraging:
  - Tree-height reduction
  - On-chip block RAMs

With that, indeed a considerable increase in the performance was achieved as can be seen by comparing the reports from the "BASE" compilation (basic optimization via PIPELINE II=1 at the second inner loop) with the report from "OPTIMIZED" (with Tree-height reduction and on-chip block RAM).


As a tradeoff, this implementation brings more complexity in the code also demands more storage in the device having as an outcome a better parallelism (DSP) for the arithmetic operations, because of Tree-height reduction reduces the dependency chain in accumulation; and throughput for data logistic, due to the fact that on-chip BRAMs allows much faster access than off-chip memory.

The increase in the performance can be spot in the values mainly for the metrics:
  - Latency (cycles): 29,844 (Reduced by ~38%).
  - Initiation Interval (II): 1 (Full pipelining achieved of critical loops).
  - Slack: 0.12 (Timing constraints met).




### Future Work
It may also be worth the effort to explore __Fixed-Point__ data types, as can be understood in the book "FPGA for Software Programmer" regarding "Hardware-Specific Optimizations". 

Since Fixed-point operations are computationally simpler and require fewer resources, the hipothesis is that it could also speedup the computation.

The eligible variables for the test would be "buffer" and the accumulators given the data logistc and number of arithmetic operations performed over them.



#### Fixed-Point data types optimization

The following statistics does not confirm any relevant increase in the performance only by converting the float into Fixed-Point.

Nevertheless, the benefit for best usage of resources is relevant as can be seen mainly in the DSP, BRAM and LUT metrics.

| Modules & Loops                                     | Issue Type | Slack | Latency (cycles) | Latency (ns) | Iteration Latency | Interval | Trip Count | Pipelined | BRAM       | DSP       | FF          | LUT         | URAM |
|-----------------------------------------------------|------------|-------|------------------|--------------|-------------------|----------|------------|-----------|------------|-----------|-------------|-------------|------|
| + func                                              | -          | 0.12  | 29828            | 1.491e+05    | -                 | 29829    | -          | no        | 18 (~0%)   | 10 (~0%)  | 1399 (~0%)  | 2675 (~0%)  | -    |
|   + func_Pipeline_VITIS_LOOP_19_3_VITIS_LOOP_20_4   | -          | 0.12  | 9621             | 4.810e+04    | -                 | 9621     | -          | no        | -          | 10 (~0%)  | 1091 (~0%)  | 1915 (~0%)  | -    |
|     o VITIS_LOOP_19_3_VITIS_LOOP_20_4              | -          | 4.25  | 9619             | 4.810e+04    | 17                | 1        | 9604       | yes       | -          | -         | -           | -           | -    |
| o VITIS_LOOP_12_1                                   | -          | 4.25  | 20200            | 1.010e+05    | 202               | -        | 100        | no        | -          | -         | -           | -           | -    |
|   o VITIS_LOOP_13_2                                 | -          | 4.25  | 200              | 1.000e+03    | 2                 | -        | 100        | no        | -          | -         | -           | -           | -    |



### Resources
  - AWS for FPGA: https://github.com/aws/aws-fpga
  - FPGA for Software Programmer: Pages(40, 41, 46, 133)