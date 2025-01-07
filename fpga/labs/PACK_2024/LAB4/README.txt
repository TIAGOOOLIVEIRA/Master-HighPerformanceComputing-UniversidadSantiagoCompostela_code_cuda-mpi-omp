With this example we learn that it is not always enough with pragmas to optimize code. It is usually needed to re-think the algorithm to fit FPGA computing. 

Try first with dot_pr.cpp. Then replace it with enhanced_dot_pr.cpp and compare both. In the second case we buffer the accum variable to achieve II=1. 

When we change the code from an initial version it is a good idea to use a testbench to be sure that the changes do not affect the functionality. Here we include an example of testbench.

Note: conditional statements are not too costly in hardware (just a bifurcation in the datapath). So do not abuse them but do not worry much about it. 

