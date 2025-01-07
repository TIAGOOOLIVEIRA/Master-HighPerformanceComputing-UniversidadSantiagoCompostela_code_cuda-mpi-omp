The aim of these LABs is to learn how to design FPGA kernels taking into account the fundamentals addressed in the lectures:

1. Vertical parallelism (pipelining)
2. Horizontal parallelism (unrolling)
3. Memory management (array partitioning)


As default we compile using AMD Vitis 2023.2. available in the CESGA. 

Run Vitis in command line mode (see "vitis_cesga.html" for details): 
v++ -c --mode hls --config hls_config.cfg --work_dir SOL1

 
