#include <iostream>
#include <vector>
#include <functional>
#include <limits>
#include <memory>
#include <atomic>
#include <algorithm>
#include <thread>
#include <future>
#include <chrono>
#include <random>
#include <unordered_set>
#include <tbb/tbb.h>

/*
clang++ -std=c++17 -I/opt/homebrew/include -L/opt/homebrew/lib -ltbb -o simple_unbalanced_tree_generator  simple_unbalanced_tree_generator.cpp

unbalanced tree: inserts nodes without rotations
insertion order: defines shape of the tree
The generateTree() inserts values sequentially
Inorder traversal prints values in sorted order, regardless of tree shape.

#TODO
    create a cmake file to compile in HPC and MacOS
    add makefile with tests to benchmark both findValidMax* alternatives for different sizes
*/

/*
std::shared_ptr ensures safe memory access during concurrent traversals, even if a node is being visited in parallel by multiple threads.
std::async and std::thread copy arguments, and std::unique_ptr is non-copyable → you'd need to pass raw pointers or references.
*/

struct Node_t {
    int value;
    std::shared_ptr<Node_t> left;
    std::shared_ptr<Node_t> right;

    explicit Node_t(int val) : value(val), left(nullptr), right(nullptr) {}
};

//Inserts a value into an unbalanced binary tree
void insert(std::shared_ptr<Node_t>& root, int value) {
    if (!root) {
        root = std::make_unique<Node_t>(value);
        return;
    }
    if (value < root->value) insert(root->left, value);
    else insert(root->right, value);
}

//Tree generator from vector
std::shared_ptr<Node_t> generateTree(const std::vector<int>& values) {
    std::shared_ptr<Node_t> root = nullptr;
    for (int val : values) insert(root, val);
    return root;
}

//TBB-parallelized generic findValidMax
int findValidMaxTBB(const std::shared_ptr<Node_t>& root,
                    const std::vector<std::function<bool(int)>>& conditions,
                    size_t depth_cutoff = 3) {
    std::atomic<int> maxValid{std::numeric_limits<int>::min()};

    std::function<void(const std::shared_ptr<Node_t>&, size_t)> visit;
    visit = [&](const std::shared_ptr<Node_t>& node, size_t depth) {
        if (!node) return;

        if (std::all_of(conditions.begin(), conditions.end(),
                        [&](const auto& cond) { return cond(node->value); })) {
            int curr = node->value;
            int prev = maxValid.load();
            while (curr > prev && !maxValid.compare_exchange_weak(prev, curr)) {}
        }

        if (depth < depth_cutoff) {
            tbb::parallel_invoke(
                [&] { visit(node->left, depth + 1); },
                [&] { visit(node->right, depth + 1); }
            );
        } else {
            visit(node->left, depth + 1);
            visit(node->right, depth + 1);
        }
    };

    visit(root, 0);
    return maxValid.load() == std::numeric_limits<int>::min() ? -1 : maxValid.load();
}


//std::thread-based optimistic lock-free findValidMax
int findValidMaxOptimistic(const std::shared_ptr<Node_t>& root, const std::vector<std::function<bool(int)>>& conditions) {
    std::atomic<int> maxValid{std::numeric_limits<int>::min()};

    std::function<void(std::shared_ptr<Node_t>)> visit;
    visit = [&](std::shared_ptr<Node_t> node) {
        if (!node) return;

        if (std::all_of(conditions.begin(), conditions.end(), [&](const auto& cond) { return cond(node->value); })) {
            int curr = node->value;
            int prev = maxValid.load();
            while (curr > prev && !maxValid.compare_exchange_weak(prev, curr)) {}
        }

        std::future<void> left, right;
        if (node->left)
            left = std::async(std::launch::async, visit, node->left);
        if (node->right)
            right = std::async(std::launch::async, visit, node->right);

        if (left.valid()) left.get();
        if (right.valid()) right.get();
    };

    visit(root);
    return maxValid.load() == std::numeric_limits<int>::min() ? -1 : maxValid.load();

}

//std::thread-based findValidMax
int findValidMaxThreads(const std::shared_ptr<Node_t>& root,
                        const std::vector<std::function<bool(int)>>& conditions) {
    std::atomic<int> maxValid{std::numeric_limits<int>::min()};
    std::mutex mtx;

    std::function<void(const std::shared_ptr<Node_t>&)> visit;
    visit = [&](const std::shared_ptr<Node_t>& node) {
        if (!node) return;

        if (std::all_of(conditions.begin(), conditions.end(),
                        [&](const auto& cond) { return cond(node->value); })) {
            std::scoped_lock lock(mtx);
            if (node->value > maxValid) maxValid = node->value;
        }

        std::future<void> left, right;
        if (node->left)
            left = std::async(std::launch::async, visit, std::cref(node->left));
        if (node->right)
            right = std::async(std::launch::async, visit, std::cref(node->right));

        if (left.valid()) left.get();
        if (right.valid()) right.get();
    };

    visit(root);
    return maxValid.load() == std::numeric_limits<int>::min() ? -1 : maxValid.load();
}

//to print the tree inorder (for testing)
void inorder(const std::shared_ptr<Node_t>& node) {
    if (!node) return;
    inorder(node->left);
    std::cout << node->value << " ";
    inorder(node->right);
}

std::vector<int> generateUniqueRandomValues(int size, int start, int end) {
    std::unordered_set<int> unique_values;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dist(start, end);

    while (unique_values.size() < static_cast<size_t>(size)) {
        unique_values.insert(dist(gen));
    }
    return std::vector<int>(unique_values.begin(), unique_values.end());
}

int main(int argc, char* argv[]) {
    if (argc < 5) {
        std::cerr << "Usage: " << argv[0] << " <size> <start> <end> <print_inorder (0|1)>\n";
        return 1;
    }

    int size = std::stoi(argv[1]);
    int start = std::stoi(argv[2]);
    int end = std::stoi(argv[3]);
    bool print_inorder = std::stoi(argv[4]) != 0;

    std::vector<int> values = generateUniqueRandomValues(size, start, end);
    auto tree = generateTree(values);

    if (print_inorder) {
        std::cout << "Inorder traversal: ";
        inorder(tree);
        std::cout << "\n";
    }

    std::vector<std::function<bool(int)>> conditions = {
        [](int x) { return x % 2 == 0; },
        [](int x) { return x > 5; }
    };

    auto start1 = std::chrono::high_resolution_clock::now();
    int result_tbb = findValidMaxTBB(tree, conditions);
    auto end1 = std::chrono::high_resolution_clock::now();

    auto start2 = std::chrono::high_resolution_clock::now();
    int result_threads = findValidMaxThreads(tree, conditions);
    auto end2 = std::chrono::high_resolution_clock::now();


    auto start3 = std::chrono::high_resolution_clock::now();
    int result_optimistic = findValidMaxOptimistic(tree, conditions);
    auto end3 = std::chrono::high_resolution_clock::now();

    std::cout << "[TBB] Max: " << result_tbb << " | Time: "
              << std::chrono::duration_cast<std::chrono::microseconds>(end1 - start1).count() << "us\n";

    std::cout << "[Threads] Max: " << result_threads << " | Time: "
              << std::chrono::duration_cast<std::chrono::microseconds>(end2 - start2).count() << "us\n";

    std::cout << "[Optimistic Threads] Max: " << result_optimistic << " | Time: "
              << std::chrono::duration_cast<std::chrono::microseconds>(end3 - start3).count() << "us\n";

    return 0;
}
