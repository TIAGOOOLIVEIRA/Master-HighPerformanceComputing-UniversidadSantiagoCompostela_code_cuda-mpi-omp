// Program to print path from root node to destination node
// for 15-puzzle algorithm using a guided Branch and Bound

#include "15puzzle.h"

#include <bits/stdc++.h>
#include <algorithm>
#include <random>

using namespace std;

#define SOL_STEPS 12

default_random_engine rng_engine;

void initialize_rng(int rng_seed)
{
  rng_engine = default_random_engine(rng_seed);
}

puzzle_t final_state =
	{
		  1,  2,  3,  4,
		  5,  6,  7,  8,
		  9, 10, 11, 12,
		 13, 14, 15,  0
	};
	
puzzle_t partial_states[12] =
  {
    { 1, 0, 0, 0,      0, 0, 0, 0,      0,  0,  0, 0,      0, 0, 0, 0 },    // first move
    { 1, 2, 0, 0,      0, 0, 0, 0,      0,  0,  0, 0,      0, 0, 0, 0 },    // second move
    { 1, 2, 4, 0,      0, 0, 0, 0,      0,  0,  0, 0,      0, 0, 0, 0 },    // third move
    { 1, 2, 4, 0,      0, 0, 3, 0,      0,  0,  0, 0,      0, 0, 0, 0 },    // prepare first row
    { 1, 2, 3, 4,      5, 0, 0, 0,      0,  0,  0, 0,      0, 0, 0, 0 },    // first row
    { 1, 2, 3, 4,      5, 6, 0, 0,      0,  0,  0, 0,      0, 0, 0, 0 },
    { 1, 2, 3, 4,      5, 6, 8, 0,      0,  0,  0, 0,      0, 0, 0, 0 },
    { 1, 2, 3, 4,      5, 6, 8, 0,      0,  0,  7, 0,      0, 0, 0, 0 },
    { 1, 2, 3, 4,      5, 6, 7, 8,      13, 9,  0, 0,      0, 0, 0, 0 },    // second row
    { 1, 2, 3, 4,      5, 6, 7, 8,      9, 14, 10, 0,      13, 0, 0, 0 },
    { 1, 2, 3, 4,      5, 6, 7, 8,      9, 10,  0, 0,      13, 14, 0, 0 },  // almost there
    { 1, 2, 3, 4,      5, 6, 7, 8,      9, 10, 11, 12,     13, 14, 15, 0 }  // final state
   };
    
// state space tree nodes


int getInvCount(int arr[])
{
	int inv_count = 0;
	for (int i = 0; i < N * N - 1; i++)
	{
		for (int j = i + 1; j < N * N; j++)
		{
			// count pairs(arr[i], arr[j]) such that 
			// i < j but arr[i] > arr[j]
			if (arr[j] && arr[i] && arr[i] > arr[j])
				inv_count++;
		}
	}
	return inv_count;
}

int findBlankPosition(puzzle_t puzzle, int &x, int &y)
{
	// start from bottom-right corner of matrix
	for (int i = N - 1; i >= 0; i--)
		for (int j = N - 1; j >= 0; j--)
			if (puzzle[i*N+j] == 0)
			{
			  x = i;
			  y = j;
			  return 1;
			}
	return 0;
}

int findBlankPosition(int puzzle[N*N], int &x, int &y)
{
	// start from bottom-right corner of matrix
	for (int i = N - 1; i >= 0; i--)
		for (int j = N - 1; j >= 0; j--)
			if (puzzle[i*N+j] == 0)
			{
			  x = i;
			  y = j;
			  return 1;
			}
	return 0;
}

int findXPosition(int puzzle[N*N])
{
	// start from bottom-right corner of matrix
	for (int i = N - 1; i >= 0; i--)
		for (int j = N - 1; j >= 0; j--)
			if (puzzle[i*N+j] == 0)
				return N - i;
	return -1;
}

bool isSolvable(puzzle_t puzzle)
{
  int * puzzle_array = &puzzle[0];
  
	// Count inversions in given puzzle
	int invCount = getInvCount((int*)puzzle_array);

	// If grid is odd, return true if inversion
	// count is even.
	if (N & 1)
		return !(invCount & 1);

	else	 // grid is even
	{
		int pos = findXPosition(puzzle_array);
		if (pos & 1)
			return !(invCount & 1);
		else
			return invCount & 1;
	}
}

// Print the sequence of states from root to lastNode
void printSolution(Node * lastNode)
{
	Node * n = lastNode;
	
	do 
	{
		cout << "[ Solution ]";
		printMatrix(n->mat);
		cout << endl;
		n = n->parent;
	} while (n);
}

void setSolutionHashes(Node * lastNode, vector<size_t> & solution_steps)
{
	Node * n = lastNode;
	solution_steps.reserve(n->level);
	do 
	{
		solution_steps.insert(solution_steps.begin(), n->hash);
		n = n->parent;
	} while (n);
}

// Function to print N x N matrix
void printMatrix(int mat[N*N])
{
  for (int i = 0; i < N*N; i++)
  {
    if (!(i%N))
      printf("| ");
    printf("%2d ", mat[i]);
  }
  printf("|");
  return;
}

// print path from root node to destination node
void printPath(Node* root)
{
	if (root == NULL)
		return;
	printPath(root->parent);
	printMatrix(root->mat);

	printf("\n");
}

// print moves {U,D,L,R} from root node to destination node
void setSolutionMoves(Node* node, char moves[MAX_SOLUTION_LENGTH])
{
	assert(node->level < MAX_SOLUTION_LENGTH);
	memset(moves, 0, MAX_SOLUTION_LENGTH);

	int nx, ny, px, py;
	Node * curnode = node->parent;

	findBlankPosition(node->mat, nx, ny);
	for (int i = node->level-1; i>=0; i--)
	{
		findBlankPosition(curnode->mat, px, py);
		if (nx != px)
		{
			assert(ny == py);
			moves[i] = (nx < px)?'R':'L';
		}
		else
		{
			assert(ny != py);
			moves[i] = (ny < py)?'D':'U';
		}
		nx = px; ny = py;
		curnode = curnode->parent;
	}
	moves[node->level] = '\n';
}

size_t build_hash(int mat[N*N])
{
  size_t h = 0;
  
  for (int i=0; i<N*N; i++)
  {
    h |= ((size_t)mat[i]) << (((size_t)i)*4);
  }
  
  return h;
}

// Function to allocate a new node
Node* newNode(int mat[N*N], int x, int y, int newX,
			int newY, int level, Node* parent,
			int prevMove)
{
	Node* node = new Node;

	node->parent = parent;
	memcpy(node->mat, mat, sizeof node->mat);
	swap(node->mat[x*N+y], node->mat[newX*N+newY]);
	node->cost = INT_MAX;
	node->level = level;
	node->x = newX;
	node->y = newY;
	node->prevMove = prevMove; 
 	node->hash = build_hash(node->mat);

  
	return node;
}

// bottom, left, top, right
int row[] = { 1, 0, -1, 0 };
int col[] = { 0, -1, 0, 1 };

#if COST_FUN == 0
  // Simple Hamming distance
  int calculateCost(int initial[N*N], int & solvestep)
  {
    assert(solvestep < SOL_STEPS);
    
	  int count = 0;
	  puzzle_t goal_mat = partial_states[solvestep];
	   
	  
	  
	  for (int i = 0; i < N; i++)
	  for (int j = 0; j < N; j++)
		  if (initial[i*N+j] && initial[i*N+j] != goal_mat[i*N+j])
		  count++;
	  return count;
  }
#elif COST_FUN == 1
  // Manhattan distance
  int calculateCost(int initial[N*N], int & solvestep)
  {
    assert(solvestep < SOL_STEPS);
    
	int count = 0, i, j;
	puzzle_t goal_mat = partial_states[solvestep];
	  
	for (i = 0; i < N*N; i++)
		{
		if ((!goal_mat[i]) || initial[i] == goal_mat[i]) continue;
		
		for (j = 0; j < N*N && initial[j] != goal_mat[i]; j++);
		
		// cost increases in row + col differences
		count += abs( (i/N) - (j/N) ) + abs( (i%N) - (j%N) );
		}

	return count;
  }
#else
  int calculateCost(int initial[N*N], int & solvestep)
  {
    assert(0);
    return 0;
  }
#endif

// Function to check if (x, y) is a valid matrix coordinate
int isSafe(int x, int y)
{
	return (x >= 0 && x < N && y >= 0 && y < N);
}

// Comparison object to be used to order the heap
struct comp
{
	bool operator()(const Node* lhs, const Node* rhs) const
	{
		return (lhs->cost + lhs->level) > (rhs->cost + rhs->level);
	}
};

puzzle_t new_random_state()
{
  auto random_state = final_state;
	shuffle ( random_state.begin(), random_state.end(), rng_engine );
	
	return random_state;
}

puzzle_t new_from_hash(size_t hash)
{
  puzzle_t hash_state(final_state);
  
  for (int i=0; i<N*N; i++)
  {
    int v = (hash >> (i*4)) & 0xF;
    hash_state[i] = v;
    // h |= ((size_t)mat[i]) << (((size_t)i)*4);
  }
  
  return hash_state;
}

int solve(puzzle_t initial,
		  int * path_length, int * nodes_alloc,
		  Optimality optimization_level,
		  char solution_moves[MAX_SOLUTION_LENGTH])
{
  	int x=0, y=0;
  	findBlankPosition(initial, x, y);
  	int solvestep = 0;
  
 	if (!isSolvable(initial))
  	{
		return SOLVER_NO_SOLUTION;
  	}
		
	priority_queue<Node*, vector<Node*>, comp> pq;
	set<size_t> visited_nodes_hash;
	set<Node*> visited_nodes;

	Node* root = newNode(&initial[0], x, y, x, y, 0, NULL, -1);
	root->cost = calculateCost(&initial[0], solvestep);

	// Add root to list of alive nodes;
	pq.push(root);
	*nodes_alloc = 1;
	*path_length = 0;
	visited_nodes_hash.insert(root->hash);

	while (!pq.empty())
	{
		Node* current_state = pq.top();
		pq.pop();
		(*nodes_alloc)++; 

		if (current_state->cost == 0 && solvestep == SOL_STEPS)
		{
			(*nodes_alloc) += pq.size();
			(*path_length)  = current_state->level; 

			//setSolutionHashes(current_state, solution_steps);
			setSolutionMoves(current_state, solution_moves);
			//printMoves(current_state);
			//printSolution(current_state);

			while (!pq.empty())
			{
			  Node* n = pq.top();
			  pq.pop();
			  delete n;
			}
			
			visited_nodes.insert(current_state);
			root = current_state;
			//return SOLVER_OK;
		}
		else if (*nodes_alloc > MAX_OPEN_NODES)
		{
			while (!pq.empty())
			{
				Node* n = pq.top();
				pq.pop();
				delete n;
			}
		}
		else
		{
		  for (int i = 0; i < 4; i++)
		  {
			  if (isSafe(current_state->x + row[i], current_state->y + col[i]))
			  {
				  if (current_state->prevMove == (i+2)%4)
				    continue;
				    
				  Node* child = newNode(current_state->mat, current_state->x,
							  current_state->y, current_state->x + row[i],
							  current_state->y + col[i],
							  current_state->level + 1, current_state, i);
				
				  if(visited_nodes_hash.insert(child->hash).second == false)
				  {
				    delete child;
				  }
				  else
				  {
				    child->cost = calculateCost(child->mat, solvestep);
				    
					while (solvestep < SOL_STEPS - 1)
					{
						int next_ss = solvestep + 1;
						int next_cost;
						if (next_cost = calculateCost(child->mat, next_ss) < child->cost)
						{
							child->cost = next_cost;
							solvestep++;
						}
						else
							break;
					}
				    if (child->cost == 0)
				    {
				      // increase step and clean queue
				      if (solvestep < SOL_STEPS - 1)
				      {
				        solvestep = min(solvestep + optimization_level, SOL_STEPS-1);
						child->cost = calculateCost(child->mat, solvestep);
				      }
				      else
				        solvestep++;
				      
					  while (!pq.empty())
					  {
					    Node* n = pq.top();
						pq.pop();
						delete n;
					  }
					  i = 4; // force break
				    }
				    pq.push(child);
				  }
			  }
		  }
		}
		
		visited_nodes.insert(current_state);
	}

	int rval = (solvestep == SOL_STEPS && root->cost == 0)?SOLVER_OK:SOLVER_CANNOT_SOLVE;
	
	for (Node * n : visited_nodes)
		delete n;
	
	return rval;
}

