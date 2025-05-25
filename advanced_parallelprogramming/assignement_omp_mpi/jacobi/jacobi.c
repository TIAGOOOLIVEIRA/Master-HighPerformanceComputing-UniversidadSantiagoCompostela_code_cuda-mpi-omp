
/************************************************************
* program to solve a finite difference 
* discretization of Helmholtz equation :  
* (d2/dx2)u + (d2/dy2)u - alpha u = f 
* using Jacobi iterative method. 
*
* Modified: Sanjiv Shah,       Kuck and Associates, Inc. (KAI), 1998
* Author:   Joseph Robicheaux, Kuck and Associates, Inc. (KAI), 1998
*
* 
* Input :  n - grid dimension in x direction 
*          m - grid dimension in y direction
*          alpha - Helmholtz constant (always greater than 0.0)
*          tol   - error tolerance for iterative solver
*          relax - Successice over relaxation parameter
*          mits  - Maximum iterations for iterative solver
*
* On output 
*       : u(n,m) - Dependent variable (solutions)
*       : f(n,m) - Right hand side function 
*************************************************************/

#include <stdio.h>
#include <math.h>

#define m 4000
#define n 4000
#define mits 100
#define tol 0.0000000001
#define alpha 0.0
#define relax 0.1
#define PI  3.1415926

void initialize(int l, int t, double al, double *dx, double *dy, double u[n][m], double f[n][m]);
void jacobi(int l, int t, double dx, double dy, double al, double omega, double u[n][m], double f[n][m], double uold[n][m],double tolerance, double maxits);
void error_check(int l, int t, double al, double *dx, double *dy, double u[][m], double f[][m]);

//double u[n][m],f[n][m],dx,dy;
//double uold[n][m];



int main()
{
    // Le metí el static para que no diese un fallo de segmento
    static double u[n][m],f[n][m];
    double dx,dy;
    static double uold[n][m];

/* Initialize data*/

    initialize (n,m,alpha,&dx,&dy,u,f);

/* Solve Helmholtz equation*/

    // double start = omp_get_wtime();
    jacobi (n,m,dx,dy,alpha,relax,u,f,uold,tol,mits);
    //double elapsed = omp_get_wtime() - start;
    
/* Check error between exact solution*/

    error_check (n,m,alpha,&dx,&dy,u,f);

}

void initialize(int l, int t, double al, double *dx, double *dy, double u[n][m], double f[n][m])
{
/******************************************************
* Initializes data 
* Assumes exact solution is u(x,y) = (1-x^2)*(1-y^2)
*
******************************************************/
  
    int i,j, xx,yy;

    *dx = 2.0 / (l-1);
    *dy = 2.0 / (t-1);
    
/* Initilize initial condition and RHS*/

    for(i=0;i<l;i++)
        for(j=0;j<t;j++){
            xx = -1.0 + (*dx) * (double)(i-1);        /* -1 < x < 1*/
            yy = -1.0 + (*dy) * (double)(j-1);        /* -1 < y < 1*/
            u[i][j] = 0.0;
            f[i][j] = -al *(1.0-xx*xx)*(1.0-yy*yy)-2.0*(1.0-xx*xx)-2.0*(1.0-yy*yy);}
    
}

void jacobi(int l, int t, double dx, double dy, double al, double omega, double u[n][m], double f[n][m], double uold[n][m],double tolerance, double maxits){
/******************************************************************
* Subroutine HelmholtzJ
* Solves poisson equation on rectangular grid assuming : 
* (1) Uniform discretization in each direction, and 
* (2) Dirichlect boundary conditions 
* 
* Jacobi method is used in this routine 
*
* Input : n,m   Number of grid points in the X/Y directions 
*         dx,dy Grid spacing in the X/Y directions 
*         alpha Helmholtz eqn. coefficient 
*         omega Relaxation factor 
*         f(n,m) Right hand side function 
*         u(n,m) Dependent variable/Solution
*         tol    Tolerance for iterative solver 
*         maxit  Maximum number of iterations 
*
* Output : u(n,m) - Solution 
*****************************************************************/

    int i,j,k;
    double error,resid,ax,ay,b;

/*
* Initialize coefficients */
    ax = 1.0/(dx*dx);  // X-direction coef
    ay = 1.0/(dy*dy); // Y-direction coef
    b  = -2.0/(dx*dx)-2.0/(dy*dy)-alpha;  // Central coeff

    error = 10.0*tol;
    k = 1;

    //while (k <= maxits && error > tolerance)
    while (k <= maxits) //We remove tolerance to fix the number of its
    {

        error = (double)0.0  ;

/* Copy new solution into old*/
        
        for(i=0;i<l;i++)
            for(j=0;j<t;j++)
                uold[i][j] = u[i][j];

/* Compute stencil, residual, & update*/

        for(i=1;i<l-1;i++)
            for(j=1;j<t-1;j++){
         
/*     Evaluate residual */
                resid = (ax*(uold[i-1][j] + uold[i+1][j]) + ay*(uold[i][j-1] + uold[i][j+1])+ b * uold[i][j] - f[i][j])/b;
/* Update solution */
                u[i][j] = uold[i][j] - omega * resid;
/* Accumulate residual error*/
                error = error + resid*resid;}
            
/* Error check */
        
        k = k + 1;

        error = (double)sqrt(error)/(double)(l*t);
    }
    
    printf("Total Number of Iterations=%d\n", k);
    printf("Residual=%E\n", error);

  }

void error_check(int l, int t, double al, double *dx, double *dy, double u[][m], double f[][m]){
/************************************************************
* Checks error between numerical and exact solution 
*
************************************************************/
      
    int i,j;
    double xx,yy,temp,error;

    *dx = 2.0 / (l-1);
    *dy = 2.0 / (t-1);
    error = 0.0;

    for(i=0;i<l;i++)
        for(j=0;j<t;j++){
            xx = -(double)1.0 + *dx * (double)(i-1);
            yy = -(double)1.0 + *dy * (double)(j-1);
            temp  = u[i][j] - (1.0-xx*xx)*(1.0-yy*yy);
            error = error + temp*temp ;
        }
  
    error = (double)sqrt(error)/(double)(l*t);

    printf("Solution Error=%E\n",error);

}
