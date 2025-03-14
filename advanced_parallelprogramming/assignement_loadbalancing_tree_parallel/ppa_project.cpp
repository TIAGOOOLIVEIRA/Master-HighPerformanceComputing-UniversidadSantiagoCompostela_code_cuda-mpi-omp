
// Remember to compile with support for at least c++11

#include <cstdio>
#include <cstdlib>
#include <climits>
#include <random>
#include <cmath>
#include <chrono>
#include <functional>
#include <vector>

static int NChildren = 2;
static int Nnodes = 0;

using mclock_t = std::chrono::high_resolution_clock;

struct Node_t {
  size_t value;
  std::vector<Node_t *> children;
  
  Node_t(int nchildren);
};

Node_t::Node_t(int nchildren) :
children(nchildren, nullptr)
{ static std::mt19937_64 gen(981);
  static std::uniform_int_distribution<size_t> dist(0, ULLONG_MAX);
  
  value = dist(gen);
  Nnodes++;
}


Node_t *build_node(int cur_level, int nlevels)
{
  const int nchildren_limit = NChildren + ceilf((nlevels - cur_level) / 2.5f);
  const int my_nchildren = (cur_level < nlevels) ? NChildren : 0;
  Node_t * const root = new Node_t(my_nchildren);

  if (my_nchildren && (root->value < (ULLONG_MAX / 5))) {
    NChildren++;
  }

  for (int i = 0; i < my_nchildren; i++) {
    root->children[i] = build_node(cur_level + 1, nlevels);
  }
  
  if (NChildren > nchildren_limit) {
    NChildren = nchildren_limit;
  }

  return root;
}

Node_t *generateTree(int nlevels)
{
  return build_node(0, nlevels);
}

size_t findValidMax(std::vector<int>& path, Node_t *root, const std::vector<std::function<bool(size_t)>>& conditions)
{ std::vector<int> local_path, best_path;

  size_t max = root->value;

  // Check all conditions are fulfilled by the value or store 0 otherwise
  //TODO
    //Parallelize with with TBB 
    //Analyze with Vtune

    //1 Jumping Right In_ Hello TBB _ Pro TBB_ C++ Parallel Programming with Threading Building Blocks.pdf
    //c_c-data-structures-and-algorithms-in-c.pdf
        //13.2.2 The Adjacency List Structure

  for (const std::function<bool(size_t)>& condition : conditions) {
    if (!condition(root->value)) {
      max = 0;
      break;
    }
  }

  for (int i = 0; i < root->children.size(); i++) {
    local_path.push_back(i);
    size_t tmp = findValidMax(local_path, root->children[i], conditions);
    if (tmp > max) {
      max = tmp;
      best_path = local_path;
    }
    local_path.clear();
  }
  
  path.insert(path.end(), best_path.begin(), best_path.end());
  
  return max;
}

size_t getPathVal(const std::vector<int>& path, Node_t *root)
{
  if (path.empty()) {
    return root->value;
  } else {
    std::vector<int> tmp(path.begin() + 1, path.end());
    return getPathVal(tmp, root->children[path[0]]);
  }
}

int main(int argc, char** argv)
{ mclock_t::time_point t0, t1;
  
  t0 = mclock_t::now();
  Node_t * const root = generateTree(7);
  t1 = mclock_t::now();

  printf("Nnodes=%d Creation time=%lf\n", Nnodes, std::chrono::duration<double>(t1 - t0).count());
  
  std::vector<int> path;
  std::vector<std::function<bool(size_t)>> conditions = {
    [] (size_t v) { return (v / 3.14159) > 8930.679; },
    [] (size_t v) { return (v + 3) != 0; },
    [] (size_t v) { return (v + 3) > v; },
    [] (size_t v) { return (79890400.73 / (v + 3)) < 7534.39; },
    [] (size_t v) { return !(v % 3); },
    [] (size_t v) { return v < 1200000000000; },
    [] (size_t v) { return (v & 0xfff00) < 0xff000; }
  };

  t0 = mclock_t::now();
  const size_t max = findValidMax(path, root, conditions);
  t1 = mclock_t::now();

  printf("Search Time=%lf\n", std::chrono::duration<double>(t1 - t0).count());
  printf("Value=%zu\n", max);

  const size_t tmp = getPathVal(path, root);
  printf("Test %s\n", (max == tmp) ? "OK" : "FAILED");

  return 0;
}

