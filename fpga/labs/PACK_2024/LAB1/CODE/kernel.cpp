#include "ap_int.h"

typedef ap_int<1024> dt; // Try to change the data type and see the effect on the execution time and resources

void func(const dt A, const dt B, dt *RES)
{
	*RES=A*B;;
}
