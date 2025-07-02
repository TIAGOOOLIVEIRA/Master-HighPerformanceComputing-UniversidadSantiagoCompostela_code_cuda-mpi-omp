#include "15puzzle.h"
#include <iostream>
#include <random>
#include <string>
#include <climits>
#include <vector>
#include <algorithm>

#define UNSOLVABLE_PROB  0.3

using namespace std;

int main(int argc, char * argv[])
{
    int puzzle_count   = 0;
    int solvable_count = 0;
    vector<size_t> puzzles;

    if (argc < 4)
    {
        cerr << "Call: " << argv[0] << " n_puzzles out_file_txt out_file_bin [rng_seed]" << endl;
        return 1;
    }

    int n_puzzles = stoi(argv[1]);
    FILE * ftxt = fopen(argv[2], "w");
    FILE * fbin = fopen(argv[3], "w");
    int rng_seed = (argc>4)?stoi(argv[4]):12345;

    default_random_engine rng = default_random_engine(rng_seed);
    uniform_real_distribution<double> distribution(0.0,1.0);

    puzzles.reserve(n_puzzles);

    while (puzzle_count < n_puzzles)
    {
        auto random_state = new_random_state();
        auto solvable = isSolvable(random_state);
        if ( solvable || (distribution(rng) < UNSOLVABLE_PROB) )
        {
            size_t random_hash = build_hash(&random_state[0]);
            puzzle_count++;
            solvable_count += solvable;

            puzzles.emplace_back(random_hash);           
        }
    }

    sort(puzzles.begin(), puzzles.end(), greater<size_t>()); 
    /* write results */
    for (size_t puzzle_hash : puzzles)
    {
        fprintf(ftxt, "%lu \n", puzzle_hash);
        fwrite(&puzzle_hash, sizeof(size_t), 1, fbin);
    }
    fprintf(ftxt, "\n");

    fclose(ftxt);
    fclose(fbin);

    float solvable_ratio = 1.0*solvable_count/puzzle_count;
    cout << "Generated   " << puzzle_count << " puzzles" << endl;
    cout << "Solvable:   " << solvable_count << " : " << solvable_ratio << endl;
    cout << "Unsolvable: " << puzzle_count - solvable_count << " : " << 1.0 - solvable_ratio << endl;
    return 0;
}