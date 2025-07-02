/*
 * Exercise of dynamic load balancing
 *
 * This program is a 15-puzzle solver (google 15-puzzle if you don't know what it is)
 * Each 15-puzzle initial state, a set of 15 tiles placed into a 4-by-4 grid, is encoded as a "size_t" hash
 * The program reads a file with a set of hashes and attempts to solve the puzzles.
 * Note that not every initial configuration is solvable.
 * 
 * The goal of this exercise is to modify this source code to use MPI-IO and one-sided operations.
 * Analyze the workload carefully and reason the best solution in order to improve the performance
 * for every potential configuration.
 *
 * This code depends on aux_files/15_puzzle.{h, cpp} which contains the actual solver.
 * The optimality of the solver can be configured when calling the solve() function in line 112.
 * Try there different values to check if the workload can be balanced correctly. 
 * 
 * Compile: mpic++ -O3 -o final_15puzzle final_15puzzle.cpp aux_files/15puzzle.cpp -lm
 * Run: mpirun -n NP ./final_15puzzle data/puzzles.100.bin
 */

 #include "aux_files/15puzzle.h"

 #include <mpi.h>
 #include <cstdio>
 #include <cstdlib>
 #include <chrono>
 #include <iomanip>
 #include <algorithm>
 #include <cmath>
 
 using namespace std;
 using namespace std::chrono;
 
 #define PUZZLE_COUNT 1000
 #define DEBUG_LEVEL 2
 
 string puzzles_filename = "puzzles.in";
 int numfill = ceil(log10(PUZZLE_COUNT));
 
 int main(int argc, char *argv[])
 {
   int mpi_rank, mpi_size;
   
   if (MPI_Init(&argc, &argv) != MPI_SUCCESS)
   {
     cerr << "Error initializing MPI" << endl;
     return 1;
   }
   
   MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
   MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
   
   FILE * input_fp, * output_fp;
   size_t next_hash;
   int path_length, nodes_alloc;
   int puzzle_count = 0;
   size_t * puzzles;
   int l_puzzle_count = 0;
 
   if (!mpi_rank)
   {
     input_fp = fopen(argv[1], "r");
     if (!input_fp)
     {
       cerr << "Cannot read input file " << argv[1] << endl;
       MPI_Abort(MPI_COMM_WORLD, 1);
     }
 
     fseek(input_fp, 0L, SEEK_END);
     long file_size = ftell(input_fp);
 
     if (file_size % sizeof(size_t))
     {
       cerr << "Wrong file size: " << file_size << endl;
       MPI_Abort(MPI_COMM_WORLD, 2);
     }
 
     fseek(input_fp, 0L, SEEK_SET);
 
     puzzle_count = file_size / sizeof(size_t);
     puzzles = (size_t *) malloc( sizeof(size_t) * puzzle_count );
 
     cout << "Will read " << puzzle_count << " puzzles" << endl;
     fread(puzzles, sizeof(size_t), puzzle_count, input_fp);
     
     fclose(input_fp);
   }
 
   MPI_Bcast(&puzzle_count, 1, MPI_INT, 0, MPI_COMM_WORLD);
   l_puzzle_count = puzzle_count / mpi_size;
 
   if (mpi_rank)
     puzzles = (size_t *) malloc( sizeof(size_t) * l_puzzle_count );
   
   MPI_Scatter(puzzles, l_puzzle_count, MPI_LONG,
               mpi_rank?puzzles:MPI_IN_PLACE, l_puzzle_count, MPI_LONG,
               0, MPI_COMM_WORLD);
 
   initialize_rng(42 + mpi_rank);
   
   auto ps_time = high_resolution_clock::now();
   
   for (int i=0; i<l_puzzle_count; i++)
   {
     auto s_time = high_resolution_clock::now();
     auto random_state = new_random_state();
    
     auto hash_state = new_from_hash(puzzles[i]);
 
     char solution_moves[MAX_SOLUTION_LENGTH];
 
     //TODO: Try different solver optimality levels
     /* optimality can be: OPT_{VERY_GOOD, GOOD, MEH, BAD, AWFUL} */
     int can_solve = solve(hash_state, &path_length, &nodes_alloc,
                           OPT_MEH,
                           solution_moves);
 
     auto e_time = high_resolution_clock::now();
     auto local_duration = duration_cast<microseconds>(e_time - s_time);
 
 #if (DEBUG_LEVEL > 0)
     cout << "[" << setfill('0') << setw(numfill) << i << "] " << setfill(' ');
 #if (DEBUG_LEVEL > 1)
     printMatrix(&hash_state[0]);
 #endif
     cout << setw(9) << setprecision(2) << fixed << local_duration.count()/1000.0 << " ms. ";
              
     switch (can_solve)
     {
       case SOLVER_OK:
         cout << "Solved with depth " << path_length << " and " << nodes_alloc << " nodes allocated" << endl;
         break;
       case SOLVER_NO_SOLUTION:
         cout << "No solution" << endl;
         break;
       case SOLVER_CANNOT_SOLVE:
         cout << "Solution unreachable" << endl;
         break;
      }
 #endif
     }
 
     auto pe_time = high_resolution_clock::now();
     auto pduration = duration_cast<microseconds>(pe_time - ps_time);
     
     MPI_Barrier(MPI_COMM_WORLD);
     
     double * runtimes = 0;
     double rtime = pduration.count()/1000.0;
 
     if (!mpi_rank)
     {
       double min_rtime = 1e+20, max_rtime = 0;
       runtimes = new double[mpi_size];
       MPI_Gather(&rtime, 1, MPI_DOUBLE, runtimes, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
       max_rtime = *max_element(runtimes, runtimes + mpi_size);
       min_rtime = *min_element(runtimes, runtimes + mpi_size);
       
       cout << endl << "Workload balance:" << endl;
       for (int i=0; i<mpi_size; ++i)
       {
         cout << "  Process " << setw(2) << i << " " << setprecision(2) << fixed << setw(10) << runtimes[i] << " ms" << endl;
       }
       delete [] runtimes;
       
       cout << endl << "  Min: " << setprecision(2) << fixed << min_rtime << ", Max: " << setprecision(2) << fixed << max_rtime << endl;
       cout << "  Balance: " << 100 - 100.0*(max_rtime - min_rtime)/max_rtime << "%" << endl;
     }
     else
     {
       MPI_Gather(&rtime, 1, MPI_DOUBLE, runtimes, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
     }
 
     MPI_Finalize();
     
     cout << endl;
     return 0;
 }
 