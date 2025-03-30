#include <mutex>
#include <iostream>
#include <vector>
#include <memory>
#include <functional>
#include <limits>
#include <tbb/tbb.h>
#include <atomic>
#include <cmath>
#include <random>
#include <chrono>
#include <climits>


/*
clang++ -std=c++17 -I/opt/homebrew/include -L/opt/homebrew/lib -ltbb -o ppa_project_par ppa_project_par.cpp
OMP_NUM_THREADS=8
compute -c 1 --mem 64

*/

static int NChildren = 2;
static int Nnodes = 0;

struct Node_t {
    size_t value;
    std::vector<std::shared_ptr<Node_t>> children;
    explicit Node_t(int nchildren);
};

Node_t::Node_t(int nchildren) : children(nchildren, nullptr) {
    static std::mt19937_64 gen(981);
    static std::uniform_int_distribution<size_t> dist(0, ULLONG_MAX);
    value = dist(gen);
    Nnodes++;
}

std::shared_ptr<Node_t> build_node(int cur_level, int nlevels) {
    const int nchildren_limit = NChildren + std::ceil((nlevels - cur_level) / 2.5f);
    const int my_nchildren = (cur_level < nlevels) ? NChildren : 0;
    std::shared_ptr<Node_t> root = std::make_shared<Node_t>(my_nchildren);

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

std::shared_ptr<Node_t> generateTree(int nlevels) {
    return build_node(0, nlevels);
}

size_t getPathVal(const std::vector<int>& path, const std::shared_ptr<Node_t>& root) {
    std::shared_ptr<Node_t> current = root;
    for (int index : path) {
        if (!current || index >= static_cast<int>(current->children.size())) return 0;
        current = current->children[index];
    }
    return current ? current->value : 0;
}

size_t findValidMax(std::vector<int>& path, const std::shared_ptr<Node_t>& root,const std::vector<std::function<bool(size_t)>>& conditions) {
    std::vector<int> local_path, best_path;

    size_t max = root->value;

    for (const auto& condition : conditions) {
        if (!condition(root->value)) {
            max = 0;
            break;
        }
    }

    for (size_t i = 0; i < root->children.size(); ++i) {
        local_path.push_back(static_cast<int>(i));
        size_t tmp = findValidMax(local_path, root->children[i], conditions);
        if (tmp > max) {
            max = tmp;
            best_path = local_path;
        }
        //semantically correct for DFS traversal.
        //local_path.pop_back
        
        local_path.clear();
    }

    path.insert(path.end(), best_path.begin(), best_path.end());

    return max;
}

size_t findValidMaxTBB_SharedPtr(std::vector<int>& path,
                                 const std::shared_ptr<Node_t>& root,
                                 const std::vector<std::function<bool(size_t)>>& conditions,
                                 size_t depth_cutoff = 3) {
    std::atomic<size_t> maxValid{0};
    std::mutex path_mutex;
    std::vector<int> best_path;

    std::function<void(const std::shared_ptr<Node_t>&, size_t, std::vector<int>)> visit;
    visit = [&](const std::shared_ptr<Node_t>& node, size_t depth, std::vector<int> local_path) {
        if (!node) return;

        if (std::all_of(conditions.begin(), conditions.end(),
                        [&](const auto& cond) { return cond(node->value); })) {
            size_t curr = node->value;
            size_t prev = maxValid.load();
            while (curr > prev && !maxValid.compare_exchange_weak(prev, curr)) {}

            if (curr > prev) {
                std::lock_guard<std::mutex> lock(path_mutex);
                best_path = local_path;
            }
        }

        if (depth < depth_cutoff) {
            tbb::parallel_for(size_t(0), node->children.size(), [&](size_t i) {
                auto next_path = local_path;
                next_path.push_back(static_cast<int>(i));
                visit(node->children[i], depth + 1, next_path);
            });
        } else {
            for (size_t i = 0; i < node->children.size(); ++i) {
                auto next_path = local_path;
                next_path.push_back(static_cast<int>(i));
                visit(node->children[i], depth + 1, next_path);
            }
        }
    };

    visit(root, 0, {});
    path = best_path;
    return maxValid.load();
}

int main() {
    using mclock_t = std::chrono::high_resolution_clock;
    mclock_t::time_point t0, t1;

    t0 = mclock_t::now();
    std::shared_ptr<Node_t> root = generateTree(7);
    t1 = mclock_t::now();
    std::cout << "Nnodes=" << Nnodes << " Creation time="
              << std::chrono::duration<double>(t1 - t0).count() << "\n";

    std::vector<std::function<bool(size_t)>> conditions = {
        [](size_t v) { return (v / 3.14159) > 8930.679; },
        [](size_t v) { return (v + 3) != 0; },
        [](size_t v) { return (v + 3) > v; },
        [](size_t v) { return (79890400.73 / (v + 3)) < 7534.39; },
        [](size_t v) { return !(v % 3); },
        [](size_t v) { return v < 1200000000000; },
        [](size_t v) { return (v & 0xfff00) < 0xff000; }
    };

    std::vector<int> path_seq;
    t0 = mclock_t::now();
    size_t max_seq = findValidMax(path_seq, root, conditions);
    t1 = mclock_t::now();
    std::cout << "[Sequential] Search Time=" << std::chrono::duration<double>(t1 - t0).count() << "\n";
    std::cout << "[Sequential] Value=" << max_seq << "\n";
    std::cout << "[Sequential] Test " << (max_seq == getPathVal(path_seq, root) ? "OK" : "FAILED") << "\n";

    std::vector<int> path_par;
    t0 = mclock_t::now();
    size_t max_par = findValidMaxTBB_SharedPtr(path_par, root, conditions);
    t1 = mclock_t::now();
    std::cout << "[Parallel] Search Time=" << std::chrono::duration<double>(t1 - t0).count() << "\n";
    std::cout << "[Parallel] Value=" << max_par << "\n";
    std::cout << "[Parallel] Test " << (max_par == getPathVal(path_par, root) ? "OK" : "FAILED") << "\n";

    return 0;
}
