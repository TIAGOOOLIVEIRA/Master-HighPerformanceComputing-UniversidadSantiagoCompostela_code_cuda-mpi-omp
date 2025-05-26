/** Version modified for SPEC Benchmark suite - see Comments below
C ***
      PROGRAM SHALOW 
C     BENCHMARK WEATHER PREDICTION PROGRAM FOR COMPARING THE
C     PREFORMANCE OF CURRENT SUPERCOMPUTERS. THE MODEL IS
C     BASED OF THE PAPER - THE DYNAMICS OF FINITE-DIFFERENCE
C     MODELS OF THE SHALLOW-WATER EQUATIONS, BY ROBERT SADOURNY
C     J. ATM. SCIENCES, VOL 32, NO 4, APRIL 1975.
C
C     CODE BY PAUL N. SWARZTRAUBER, NATIONAL CENTER FOR
C     ATMOSPHERIC RESEARCH, BOULDER, CO,  OCTOBER 1984.
C
C     MODIFIED BY R. K. SATO, NCAR, APRIL 7, 1986 FOR MULTITASKING.
C     MODIFIED FOR SPEC to run longer: ITMAX inCremented from 120 to
C                                      1200 - J.Lo 7/19/90
C     Modified by Bodo Parady for the SPECpar suite.  Various
C     compilation sizes added.  Iterations reduced to orginal
C     problem, but size increased 4x in each dimension.
C
C     Further modified by Reinhold Weicker (Siemens Nixdorf) for the
C     SPEC CFP95 suite:
C     Changed back to the form with PARAMETER statements for
C     N1 and N2, set N1 = N2 = 1335.
C     The execution time is determined by these values and the
C     variable ITMAX (number of iterations) read from the input file.
C     Execution time is linear in ITMAX.*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define N1 1335
#define N2 1335

#define min(a,b) (((a) < (b)) ? (a) : (b))

float U[N1][N2], V[N1][N2], P[N1][N2], UNEW[N1][N2], VNEW[N1][N2], PNEW[N1][N2], UOLD[N1][N2], VOLD[N1][N2], POLD[N1][N2], CU[N1][N2], CV[N1][N2], Z[N1][N2], H[N1][N2], PSI[N1][N2];

int  ITMAX, MPRINT, M, N, MP1, NP1;

float DT,TDT,DX,DY,A,ALPHA,EL,PI,TPI,DI,DJ,PCF;

void initial(void);
void calc1(void);
void calc2(void);
void calc3z(void);
void calc3(void);

int main(){
    FILE *fp;
    int NCYCLE, i, MNMIN, ICHECK, JCHECK;
    float TIME, PTIME, PCHECK, UCHECK, VCHECK;
    
    printf("SPEC benchmark 171.swim\n");
    printf("\n");
    
    if ((fp=fopen("SWIM7","w"))==NULL){
        printf("No se puede abrir el archivo\n");
        exit(1);}
    
/*C       INITIALIZE CONSTANTS AND ARRAYS*/

    initial();

/*C     PRINT INITIAL VALUES*/

    
    printf("NUMBER OF POINTS IN THE X DIRECTION %d\n", N);
    printf("NUMBER OF POINTS IN THE y DIRECTION %d\n", M);
    printf("GRID SPACING IN THE X DIRECTION %f\n", DX);
    printf("GRID SPACING IN THE Y DIRECTION %f\n", DY);
    printf("TIME STEP %f\n", DT);
    printf("TIME FILTER PARAMETER %f\n", ALPHA);
    printf("NUMBER OF ITERATIONS %d\n", ITMAX);
    
    MNMIN = min(M,N);

/*C  6/22/95 for SPEC: JWR: Initialization of TIME*/

    TIME = 0;
    NCYCLE = 0;
    
    BUCLE:    NCYCLE = NCYCLE + 1;

/*     COMPUTE CAPITAL  U, CAPITAL V, Z AND H*/
        
        calc1();

/*     COMPUTE NEW VALUES U,V AND P*/
        
        calc2();
        
        TIME = TIME + DT;
    if((NCYCLE % MPRINT) != 0) goto TESTEND;
    PTIME = TIME/3600.0;

    /*
C *** modified for SPEC results verification
C *** We want to ensure that all calculations were done
C *** so we "use" all of the computed results.
C ***
C *** Since all of the comparisons of the individual diagnal terms
C *** often differ in the smaller values, the check we have selected
C *** is to add the absolute values of all terms and print these results
C*/
    
    fprintf(fp, "\n\n%s", "CYCLE NUMBER  ");
    fprintf(fp, "%d", NCYCLE);
    fprintf(fp, "\t%s", "MODEL TIME IN  HOURS  ");
    fprintf(fp, "%f\n\n", PTIME);
    fprintf(fp, "%s\n\n", "DIAGONAL ELEMENTS OF U");
    for(i=0;i<MNMIN;i=i+10)
        fprintf(fp, "%E\t", UNEW[i][i]);
    fprintf(fp, "%s\n", " ");
    
    PCHECK = 0.0;
    UCHECK = 0.0;
    VCHECK = 0.0;
    
    for(ICHECK=0; ICHECK<MNMIN; ICHECK++){
        for(JCHECK=0; JCHECK<MNMIN; JCHECK++){
            PCHECK = PCHECK + fabs(PNEW[ICHECK][JCHECK]);
            UCHECK = UCHECK + fabs(UNEW[ICHECK][JCHECK]);
            VCHECK = VCHECK + fabs(VNEW[ICHECK][JCHECK]);}
        UNEW[ICHECK][ICHECK] = UNEW[ICHECK][ICHECK] * ( (ICHECK%100) /100.0);
    }
    printf("\n");
    printf("Pcheck = %E\nUcheck = %E\nVcheck = %E\n", PCHECK, UCHECK, VCHECK);

/*C        TEST FOR END OF RUN*/
    TESTEND:      if(NCYCLE >= ITMAX) exit(0);

/*C     TIME SMOOTHING AND UPDATE FOR NEXT CYCLE*/


    if(NCYCLE <= 1){
        calc3z();}
    else{
        calc3();}
 
    goto BUCLE;
    

}



void initial(void){
/*C        INITIALIZE CONSTANTS AND ARRAYS
C           R. K. SATO 4/7/86
C
C
C     NOTE BELOW THAT TWO DELTA T (TDT) IS SET TO DT ON THE FIRST
C     CYCLE AFTER WHICH IT IS RESET TO DT+DT.
C
C The following code  was in SWM256, however, it was replaced by 
C		READ statements to avoid calculations to be done during
C		compile time.*/

/*CMARIA LEO LOS DATOS DESDE UN FICHERO*/
    FILE *fpin;
    int i,j;
    
    if((fpin=fopen("data/swim.in.train","r"))==NULL){
        printf("No se piede abrir el archivo\n");
        exit(1);
    }
    fscanf(fpin, "%f", &DT);
    fscanf(fpin, "%f", &DX);
    fscanf(fpin, "%f", &DY);
    fscanf(fpin, "%f", &A);
    fscanf(fpin, "%f", &ALPHA);
    fscanf(fpin, "%d", &ITMAX);
    fscanf(fpin, "%d", &MPRINT);
    fscanf(fpin, "%d", &M);
    fscanf(fpin, "%d", &N);
    
    TDT = DT;
    MP1 = M+1;
    NP1 = N+1;
    EL = N*DX;
    PI = 4.0*atan(1.0);
    TPI = PI+PI;
    DI = TPI/M;
    DJ = TPI/N;
    PCF = PI*PI*A*A/(EL*EL);

/*C     INITIAL VALUES OF THE STREAM FUNCTION AND P*/
    
    for(i=0;i<MP1;i++)
        for(j=0;j<NP1;j++){
            PSI[i][j] = A*sin((i+.5)*DI)*sin((j+.5)*DJ);
            P[i][j] = PCF*(cos(2.0*(i)*DI)+cos(2.0*(j)*DJ))+50000.0;
        }

/*C     INITIALIZE VELOCITIES*/
    
    for(i=0;i<M;i++)
        for(j=0;j<N;j++){
            U[i+1][j] = -1*(PSI[i+1][j+1]-PSI[i+1][j])/DY;
            V[i][j+1] = (PSI[i+1][j+1]-PSI[i][j+1])/DX;}

/*C     PERIODIC CONTINUATION*/

    for(j=0;j<N;j++){
        U[0][j] = U[M][j];
        V[M][j+1] = V[0][j+1];}
   
    for(i=0;i<M;i++){
        U[i+1][N] = U[i+1][0];
        V[i][0] = V[i][N];}
   
    U[0][N] = U[M][0];
    V[M][0] = V[0][N];
    
    for(i=0;i<MP1;i++)
        for(j=0;j<NP1;j++){
            UOLD[i][j] = U[i][j];
            VOLD[i][j] = V[i][j];
            POLD[i][j] = P[i][j];}
 
}
void calc1(void){
/*C        COMPUTE CAPITAL  U, CAPITAL V, Z AND H*/

    float FSDX, FSDY;
    int I,J;
    
    FSDX = 4.0/DX;
    FSDY = 4.0/DY;

    for (I=0;I<M;I++)
        for(J=0;J<N;J++){
            CU[I+1][J] = .5*(P[I+1][J]+P[I][J])*U[I+1][J];
            CV[I][J+1] = .5*(P[I][J+1]+P[I][J])*V[I][J+1];
            Z[I+1][J+1] = (FSDX*(V[I+1][J+1]-V[I][J+1])-FSDY*(U[I+1][J+1]-U[I+1][J]))/(P[I][J]+P[I+1][J]+P[I+1][J+1]+P[I][J+1]);
                           H[I][J] = P[I][J]+.25*(U[I+1][J]*U[I+1][J]+U[I][J]*U[I][J]+V[I][J+1]*V[I][J+1]+V[I][J]*V[I][J]);}
   

/*C     PERIODIC CONTINUATION*/
          
    for(J=0;J<N;J++){
        CU[0][J] = CU[M][J];
        CV[M][J+1] = CV[0][J+1];
        Z[0][J+1] = Z[M][J+1];
        H[M][J] = H[0][J];}
  
    for(I=0;I<M;I++){
        CU[I+1][N] = CU[I+1][0];
        CV[I][0] = CV[I][N];
        Z[I+1][0] = Z[I+1][N];
        H[I][N] = H[I][0];}
                           
    CU[0][N] = CU[M][0];
    CV[M][0] = CV[0][N];
    Z[0][0] = Z[M][N];
    H[M][N] = H[0][0];
}

void calc2(void){
/*C        COMPUTE NEW VALUES OF U,V,P*/
    float TDTS8, TDTSDX,TDTSDY;
    int I,J;
    
    TDTS8 = TDT/8.0;
    TDTSDX = TDT/DX;
    TDTSDY = TDT/DY;

    for(I=0;I<M;I++)
        for(J=0;J<N;J++){
            UNEW[I+1][J] = UOLD[I+1][J]+TDTS8*(Z[I+1][J+1]+Z[I+1][J])*(CV[I+1][J+1]+CV[I][J+1]+CV[I][J]+CV[I+1][J])-TDTSDX*(H[I+1][J]-H[I][J]);
            VNEW[I][J+1] = VOLD[I][J+1]-TDTS8*(Z[I+1][J+1]+Z[I][J+1])*(CU[I+1][J+1]+CU[I][J+1]+CU[I][J]+CU[I+1][J])-TDTSDY*(H[I][J+1]-H[I][J]);
            PNEW[I][J] = POLD[I][J]-TDTSDX*(CU[I+1][J]-CU[I][J])-TDTSDY*(CV[I][J+1]-CV[I][J]);}
  
    
/*C     PERIODIC CONTINUATION*/

    for(J=0;J<N;J++){
        UNEW[0][J] = UNEW[M][J];
        VNEW[M][J+1] = VNEW[0][J+1];
        PNEW[M][J] = PNEW[0][J];}
    
    for(I=0;I<M;I++){
        UNEW[I+1][N] = UNEW[I+1][0];
        VNEW[I][0] = VNEW[I][N];
        PNEW[I][N] = PNEW[I][0];}
 
    UNEW[0][N] = UNEW[M][0];
    VNEW[M][0] = VNEW[0][N];
    PNEW[M][N] = PNEW[0][0];
}

void calc3z(void){
/*C         TIME SMOOTHER FOR FIRST ITERATION*/

    int I,J;
    
    TDT = TDT+TDT;
    
    for(I=0;I<MP1;I++)
        for(J=0;J<NP1;J++){
            UOLD[I][J] = U[I][J];
            VOLD[I][J] = V[I][J];
            POLD[I][J] = P[I][J];
            U[I][J] = UNEW[I][J];
            V[I][J] = VNEW[I][J];
            P[I][J] = PNEW[I][J];}
    
}

void calc3(void){
/*C        TIME SMOOTHING AND UPDATE FOR NEXT CYCLE*/
    int I,J;
    
    for(I=0;I<M;I++)
        for(J=0;J<N;J++){
            UOLD[I][J] = U[I][J]+ALPHA*(UNEW[I][J]-2.*U[I][J]+UOLD[I][J]);
            VOLD[I][J] = V[I][J]+ALPHA*(VNEW[I][J]-2.*V[I][J]+VOLD[I][J]);
            POLD[I][J] = P[I][J]+ALPHA*(PNEW[I][J]-2.*P[I][J]+POLD[I][J]);
            U[I][J] = UNEW[I][J];
            V[I][J] = VNEW[I][J];
            P[I][J] = PNEW[I][J];}

/*C     PERIODIC CONTINUATION*/

    for(J=0;J<N;J++){
        UOLD[M][J] = UOLD[0][J];
        VOLD[M][J] = VOLD[0][J];
        POLD[M][J] = POLD[0][J];
        U[M][J] = U[0][J];
        V[M][J] = V[0][J];
        P[M][J] = P[0][J];}
  
    for(I=0;I<M;I++){
        UOLD[I][N] = UOLD[I][0];
        VOLD[I][N] = VOLD[I][0];
        POLD[I][N] = POLD[I][0];
        U[I][N] = U[I][0];
        V[I][N] = V[I][0];
        P[I][N] = P[I][0];}
    
      UOLD[M][N] = UOLD[0][0];
      VOLD[M][N] = VOLD[0][0];
      POLD[M][N] = POLD[0][0];
      U[M][N] = U[0][0];
      V[M][N] = V[0][0];
      P[M][N] = P[0][0];
}
