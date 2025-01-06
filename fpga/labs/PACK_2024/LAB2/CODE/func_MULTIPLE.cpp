/* As default two modules will be created: one multiplier and one adder/subtractor.*/
/* Try to increase the clock period (T=40ns) => 2 multipliers, 1 adder and 1 substractor are now created. */

typedef float dt;

void func(dt const A, dt const B, dt const C, dt const D, dt const E, dt *RES)
{

	dt t1=A*B;
	dt t2=C+t1;
	dt t3=D*t2;
	*RES=t3-E;
}
