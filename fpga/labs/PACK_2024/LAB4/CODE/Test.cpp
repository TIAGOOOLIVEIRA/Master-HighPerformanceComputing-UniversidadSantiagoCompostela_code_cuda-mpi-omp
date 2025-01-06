#include "dot_pr.h"
#include <iostream>

void sw_func(const dt  A[M], const dt  B[M], dt *Y)
{
	dt accum=0;
	Loop: for(int i=0; i<M; i++)
		{
		accum+=A[i]*B[i];
		}
	*Y=accum;
}

int main() 
{
  dt A[M],B[M];
  dt sw_out, hw_out;
  
  for(int i = 0; i < M; i++)
  {
  A[i]=rand() % 100;
  B[i]=rand() % 100;
  }

  
sw_func(A,B,&sw_out);
func(A,B,&hw_out);

if (sw_out==hw_out) std::cout << "Test ran successfully.\n";
else std::cout << "Test failed!\n";

  return 0;
}
