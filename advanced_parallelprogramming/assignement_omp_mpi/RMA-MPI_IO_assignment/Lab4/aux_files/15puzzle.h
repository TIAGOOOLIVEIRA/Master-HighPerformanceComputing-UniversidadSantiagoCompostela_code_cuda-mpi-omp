#ifndef _15PUZZLE_H
#define _15PUZZLE_H

#define MAX_OPEN_NODES 4000000
#define MAX_DEPTH      150

#include <vector>
#include <cstddef>

typedef std::vector<int> puzzle_t;

typedef enum
{
  OPT_VERY_GOOD = 1,
  OPT_GOOD      = 2,
  OPT_MEH       = 3,
  OPT_POOR      = 4,
  OPT_BAD       = 5,
  OPT_AWFUL     = 6
} Optimality;

#ifndef OPTIMALITY
  // 1 : Very good
  // 2 : Quite good
  // 3 : Meh
  // 4 : Not very optimal
  // 6 : Be patient
  // >6 : Hope you have a lot of memory
  #define OPTIMALITY 1
#endif

  // 0 : Hamming distance
  // 1 : Manhattan distance
  #define COST_FUN 1

  #define N 4

  #define SOLVER_OK           0
  #define SOLVER_NO_SOLUTION  1
  #define SOLVER_CANNOT_SOLVE 2
  
  #define MAX_SOLUTION_LENGTH 150

  struct Node
  {
    Node* parent;

    int mat[N*N]; // matrix
    int x, y;      // blank tile cords
    int cost;      // misplaced tiles
    int level;     // depth
    int prevMove;  // previous move
    size_t hash;
  };

  puzzle_t new_random_state();
  
  int calculateCost(int initial[N*N], int & solvestep);
  
  bool isSolvable(puzzle_t puzzle);

  // Function to solve N*N - 1 puzzle algorithm using
  // Branch and Bound. x and y are blank tile coordinates
  // in initial state
  int solve(puzzle_t initial,
	  	      int * path_length, int * nodes_alloc,
            Optimality optimization_level,
            char solution_steps[MAX_SOLUTION_LENGTH]);
	  	  
	void printMatrix(int mat[N*N]);
  
  void initialize_rng(int rng_seed);
  
  size_t build_hash(int mat[N*N]);
  
  puzzle_t new_from_hash(size_t hash);
#endif
