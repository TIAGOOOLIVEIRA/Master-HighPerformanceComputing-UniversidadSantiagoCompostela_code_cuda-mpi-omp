# Assignment 3. Utilization of Shuﬄes Instructions

- We start from a matrix X of size N ×M ×P to which the following steps in the following
are carried out:
  - Xi,j,l​=Xi,j,l​+k=0minP​Xi,j,k​,∀i,j,l
  - Xi,j,l​=Xi,j,l​+Xi,j−1,l​+Xi,j+1,l​,∀i,j,l
  - Xi,j,l​=Xi,j,l​+Xi−1,j,l​+Xi+1,j,l​,∀i,j,l

• The initial value of the matrix is Xi,j,l= l + 1, 0 ≤l < P, ∀i, j.
• The file salida shows an example for N = 4, M = 8 and P = 16 and output for each step .

- Aspects of your work that you should consider:
  - Include NAME at the top of your write-up. A ONLY file (zip, example) by student.
Develop a sequential version in the CPU to compare the time of execution and check the
error. To detect correctness of the program, draw program has an option in order to run
the sequential version of the reference CPU then compares the resulting images to ensure
correctness.

  - Step 1 would be done using shuﬄe instructions. Communication between step 1 and step
2 would be through the Shared Memory. Step 2 and 3 would be done using two kernels
and communicating data through global memory.

  - Complete the word A3-report file, with the requested data.