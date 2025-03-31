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
#include <omp.h>


/*
To build for macOS:
    make
To build for HPC 
    make hpc
To clean:
    make clean

To reserver resources on HPC:
    compute -c 1 --mem 64
To run
    ./ppa_project_par

To profile
    module load intel vtune valgrind
    vtune -collect performance-snapshot -collect memory-access -collect hotspots -collect threading -- ./ppa_project_par
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


//Actor model with TBB + OpenMP
//This is a hybrid approach that uses TBB for task parallelism and OpenMP for thread parallelism within deep or heavy subtrees.
//The TBB task group is used to manage the parallel execution of tasks, while OpenMP is used for parallelizing the execution of tasks within the TBB task group.
//This approach intends for better load balancing and can improve performance in cases where the tree structure is uneven or has deep subtrees.
size_t findValidMaxTBB_SharedPtr(std::vector<int>& path,
    const std::shared_ptr<Node_t>& root,
    const std::vector<std::function<bool(size_t)>>& conditions) {

        struct PathWithMax {
            size_t value;
            std::vector<int> path;
        };
    
        std::atomic<size_t> maxValid{0};
        std::shared_ptr<PathWithMax> bestPathWithMax = nullptr;
    
        tbb::task_arena arena;

        arena.execute([&] {
            std::function<void(const std::shared_ptr<Node_t>&, std::vector<int>)> actor;
            actor = [&](const std::shared_ptr<Node_t>& node, std::vector<int> currentPath) -> void {
                if (!node) return;
    
                bool isValid = std::all_of(conditions.begin(), conditions.end(),
                                           [&](const auto& cond) { return cond(node->value); });
                if (isValid) {
                    size_t prev = maxValid.load();
                    while (node->value > prev && !maxValid.compare_exchange_weak(prev, node->value)) {}
    
                    if (node->value > prev) {
                        auto newPath = std::make_shared<PathWithMax>(PathWithMax{node->value, currentPath});
                        auto expected = std::atomic_load(&bestPathWithMax);
                        while ((expected == nullptr || newPath->value > expected->value) &&
                               !std::atomic_compare_exchange_weak(&bestPathWithMax, &expected, newPath)) {}
                    }
                }

                size_t childCount = node->children.size();
                if (childCount > 0) {
                    //=====actor model coordination with hybrid TBB + OpenMP=====
                    tbb::task_group tg;
                    for (size_t i = 0; i < childCount; ++i) {
                        tg.run([&, i] {
                            std::vector<int> childPath = currentPath;
                            childPath.push_back(static_cast<int>(i));
    
                            const auto& child = node->children[i];
                            if (child && child->children.size() > 4) {
                                //OpenMP inside deep or heavy subtrees
                                #pragma omp parallel
                                {
                                    #pragma omp single nowait
                                    actor(child, childPath);
                                }
                            } else {
                                //otherwise run recursively
                                actor(child, childPath);
                            }
                        });
                    }
                    tg.wait();
                }
            };
    
            actor(root, {});
        });       
        
        auto finalPath = std::atomic_load(&bestPathWithMax);
        if (finalPath) path = finalPath->path;
        return maxValid.load();

        
}


size_t findValidMaxTBB_SharedPtr_actor(std::vector<int>& path,
    const std::shared_ptr<Node_t>& root,
    const std::vector<std::function<bool(size_t)>>& conditions) {

    struct PathWithMax {
    size_t value;
    std::vector<int> path;
    };

    std::atomic<size_t> maxValid{0};
    std::shared_ptr<PathWithMax> bestPathWithMax = nullptr;

    struct Task {
        std::shared_ptr<Node_t> node;
        std::vector<int> path;
    };

    tbb::task_arena arena;
    arena.execute([&] {
        tbb::concurrent_queue<Task> taskQueue;
        taskQueue.push({root, {}});

        tbb::parallel_for(0, static_cast<int>(tbb::this_task_arena::max_concurrency()), [&](int) {
            Task task;
            while (taskQueue.try_pop(task)) {
                auto& node = task.node;
                auto currentPath = task.path;

                if (!node) continue;

                if (std::all_of(conditions.begin(), conditions.end(), [&](const auto& cond) { return cond(node->value); })) {
                    size_t prev = maxValid.load();
                    while (node->value > prev && !maxValid.compare_exchange_weak(prev, node->value)) {}

                    if (node->value > prev) {
                        auto newPath = std::make_shared<PathWithMax>(PathWithMax{node->value, currentPath});
                        auto expected = std::atomic_load(&bestPathWithMax);
                        while ((expected == nullptr || newPath->value > expected->value) &&
                               !std::atomic_compare_exchange_weak(&bestPathWithMax, &expected, newPath)) {}
                    }
                }

                //batching children
                constexpr size_t batchSize = 4;
                for (size_t i = 0; i < node->children.size(); i += batchSize) {
                    for (size_t j = i; j < std::min(i + batchSize, node->children.size()); ++j) {
                        std::vector<int> nextPath = currentPath;
                        nextPath.push_back(static_cast<int>(j));
                        taskQueue.push({node->children[j], nextPath});
                    }
                }
            }
        });
    });

    auto finalPath = std::atomic_load(&bestPathWithMax);
    if (finalPath) path = finalPath->path;
    return maxValid.load();
}

size_t findValidMaxTBB_SharedPtr6(std::vector<int>& path,
    const std::shared_ptr<Node_t>& root,
    const std::vector<std::function<bool(size_t)>>& conditions) {
    struct PathWithMax {
    size_t value;
    std::vector<int> path;
    };

    std::atomic<size_t> maxValid{0};
    std::shared_ptr<PathWithMax> bestPathWithMax = nullptr;

    tbb::task_arena arena;
    arena.execute([&] {
        std::function<void(const std::shared_ptr<Node_t>&, std::vector<int>)> rake;
        rake = [&](const std::shared_ptr<Node_t>& node, std::vector<int> currentPath) -> void {
            if (!node) return;

            bool isValid = std::all_of(conditions.begin(), conditions.end(),
                                       [&](const auto& cond) { return cond(node->value); });
            if (isValid) {
                size_t prev = maxValid.load();
                while (node->value > prev && !maxValid.compare_exchange_weak(prev, node->value)) {}

                if (node->value > prev) {
                    auto newPath = std::make_shared<PathWithMax>(PathWithMax{node->value, currentPath});
                    auto expected = std::atomic_load(&bestPathWithMax);
                    while ((expected == nullptr || newPath->value > expected->value) &&
                           !std::atomic_compare_exchange_weak(&bestPathWithMax, &expected, newPath)) {}
                }
            }
            size_t childCount = node->children.size();
            if (childCount > 0) {
                size_t batchSize = std::max(size_t(1), std::min(size_t(8), childCount / std::thread::hardware_concurrency()));
                tbb::task_group tg;
                for (size_t i = 0; i < childCount; i += batchSize) {
                    tg.run([&, i] {
                        for (size_t j = i; j < std::min(i + batchSize, childCount); ++j) {
                            std::vector<int> childPath = currentPath;
                            childPath.push_back(static_cast<int>(j));
                            rake(node->children[j], childPath);
                        }
                    });
                }
                tg.wait();
            }
        };

        rake(root, {});
    });

    auto finalPath = std::atomic_load(&bestPathWithMax);
    if (finalPath) path = finalPath->path;
    return maxValid.load();
}

size_t findValidMaxTBB_SharedPtr5(std::vector<int>& path,
    const std::shared_ptr<Node_t>& root,
    const std::vector<std::function<bool(size_t)>>& conditions) {
    
    struct PathWithMax {
    size_t value;
    std::vector<int> path;
    };

    std::atomic<size_t> maxValid{0};
    std::shared_ptr<PathWithMax> bestPathWithMax = nullptr;

    tbb::task_arena arena;
    arena.execute([&] {
        std::function<void(const std::shared_ptr<Node_t>&, std::vector<int>)> rake;
        rake = [&](const std::shared_ptr<Node_t>& node, std::vector<int> currentPath) -> void {
            if (!node) return;

            bool isValid = std::all_of(conditions.begin(), conditions.end(),
                                       [&](const auto& cond) { return cond(node->value); });
            if (isValid) {
                size_t prev = maxValid.load();
                while (node->value > prev && !maxValid.compare_exchange_weak(prev, node->value)) {}

                if (node->value > prev) {
                    auto newPath = std::make_shared<PathWithMax>(PathWithMax{node->value, currentPath});
                    auto expected = std::atomic_load(&bestPathWithMax);
                    while ((expected == nullptr || newPath->value > expected->value) &&
                           !std::atomic_compare_exchange_weak(&bestPathWithMax, &expected, newPath)) {}
                }
            }
            if (!node->children.empty()) {
                size_t batchSize = 4;
                tbb::parallel_for(size_t(0), node->children.size(), [&](size_t i) {
                    if (i % batchSize == 0) {
                        for (size_t j = i; j < std::min(i + batchSize, node->children.size()); ++j) {
                            std::vector<int> childPath = currentPath;
                            childPath.push_back(static_cast<int>(j));
                            rake(node->children[j], childPath);
                        }
                    }
                });
            }
        };

        rake(root, {});
    });
    
    auto finalPath = std::atomic_load(&bestPathWithMax);
    if (finalPath) path = finalPath->path;
    return maxValid.load();
}


size_t findValidMaxTBB_SharedPtr4(std::vector<int>& path,
    const std::shared_ptr<Node_t>& root,
    const std::vector<std::function<bool(size_t)>>& conditions) {

    struct PathWithMax {
    size_t value;
    std::vector<int> path;
    };

    std::atomic<size_t> maxValid{0};
    std::shared_ptr<PathWithMax> bestPathWithMax = nullptr;

    std::function<void(const std::shared_ptr<Node_t>&, std::vector<int>)> rake;
    rake = [&](const std::shared_ptr<Node_t>& node, std::vector<int> currentPath) -> void {
        if (!node) return;

        bool isValid = std::all_of(conditions.begin(), conditions.end(),
            [&](const auto& cond) { return cond(node->value); });
        if (isValid) {
            size_t prev = maxValid.load();
            while (node->value > prev && !maxValid.compare_exchange_weak(prev, node->value)) {}

            if (node->value > prev) {
                auto newPath = std::make_shared<PathWithMax>(PathWithMax{node->value, currentPath});
                auto expected = std::atomic_load(&bestPathWithMax);
                while ((expected == nullptr || newPath->value > expected->value) &&
                !std::atomic_compare_exchange_weak(&bestPathWithMax, &expected, newPath)) {}
            }
        }

        if (!node->children.empty()) {
            tbb::parallel_for(size_t(0), node->children.size(), [&](size_t i) {
                std::vector<int> childPath = currentPath;
                childPath.push_back(static_cast<int>(i));
                rake(node->children[i], childPath);
            });
        }
    };

    rake(root, {});

    auto finalPath = std::atomic_load(&bestPathWithMax);
    if (finalPath) path = finalPath->path;
    return maxValid.load();
}

size_t findValidMaxTBB_SharedPtr3(std::vector<int> &path,
                                 const std::shared_ptr<Node_t> &root,
                                 const std::vector<std::function<bool(size_t)>> &conditions,
                                 size_t depth_cutoff = 3)
{
    struct PathWithMax
    {
        size_t value;
        std::vector<int> path;
    };

    std::atomic<size_t> maxValid{0};
    std::shared_ptr<PathWithMax> bestPathWithMax = nullptr;

    tbb::task_arena arena;
    arena.execute([&]{
            std::function<void(const std::shared_ptr<Node_t>&, size_t, std::shared_ptr<std::vector<int>>)> visit;
            visit = [&](const std::shared_ptr<Node_t>& node, size_t depth, std::shared_ptr<std::vector<int>> local_path) {
                if (!node) return;

                if (std::all_of(conditions.begin(), conditions.end(),
                [&](const auto& cond) { return cond(node->value); })) {
                    size_t curr = node->value;
                    size_t prev = maxValid.load();
                    while (curr > prev && !maxValid.compare_exchange_weak(prev, curr)) {}

                    if (curr > prev) {
                        auto newPath = std::make_shared<PathWithMax>(PathWithMax{curr, *local_path});
                        auto expected = std::atomic_load(&bestPathWithMax);
                        while ((expected == nullptr || newPath->value > expected->value) &&
                        !std::atomic_compare_exchange_weak(&bestPathWithMax, &expected, newPath)) {}
                    }
                }

                if (depth < depth_cutoff && node->children.size() > 1) {
                tbb::task_group tg;
                for (size_t i = 0; i < node->children.size(); ++i) {
                tg.run([&, i] {
                auto next_path = std::make_shared<std::vector<int>>(*local_path);
                next_path->push_back(static_cast<int>(i));
                visit(node->children[i], depth + 1, next_path);
                });
                }
                tg.wait();
                } else {
                for (size_t i = 0; i < node->children.size(); ++i) {
                auto next_path = std::make_shared<std::vector<int>>(*local_path);
                next_path->push_back(static_cast<int>(i));
                visit(node->children[i], depth + 1, next_path);
                }
                }
                };

                visit(root, 0, std::make_shared<std::vector<int>>()); });

    auto finalPath = std::atomic_load(&bestPathWithMax);
    if (finalPath)
        path = finalPath->path;
    return maxValid.load();
}

size_t findValidMaxTBB_SharedPtr2(std::vector<int>& path,const std::shared_ptr<Node_t>& root,const std::vector<std::function<bool(size_t)>>& conditions,size_t depth_cutoff = 3) {
    
    struct PathWithMax {
    size_t value;
    std::vector<int> path;
    };

    std::atomic<size_t> maxValid{0};
    std::shared_ptr<PathWithMax> bestPathWithMax = nullptr;


    std::function<void(const std::shared_ptr<Node_t>&, size_t, std::shared_ptr<std::vector<int>>)> visit;
    visit = [&](const std::shared_ptr<Node_t>& node, size_t depth, std::shared_ptr<std::vector<int>> local_path) {
        if (!node) return;

        if (std::all_of(conditions.begin(), conditions.end(),
        [&](const auto& cond) { return cond(node->value); })) {
            size_t curr = node->value;
            size_t prev = maxValid.load();
            while (curr > prev && !maxValid.compare_exchange_weak(prev, curr)) {}

            if (curr > prev) {
                auto newPath = std::make_shared<PathWithMax>(PathWithMax{curr, *local_path});
                auto expected = std::atomic_load(&bestPathWithMax);
                while ((expected == nullptr || newPath->value > expected->value) &&
                       !std::atomic_compare_exchange_weak(&bestPathWithMax, &expected, newPath)) {}
            }
        }

        if (depth < depth_cutoff && node->children.size() > 1) {
            tbb::task_group tg;
            for (size_t i = 0; i < node->children.size(); ++i) {
                tg.run([&, i] {
                auto next_path = std::make_shared<std::vector<int>>(*local_path);
                next_path->push_back(static_cast<int>(i));
                visit(node->children[i], depth + 1, next_path);
            });
            }
            tg.wait();
        } else {
            for (size_t i = 0; i < node->children.size(); ++i) {
            auto next_path = std::make_shared<std::vector<int>>(*local_path);
            next_path->push_back(static_cast<int>(i));
            visit(node->children[i], depth + 1, next_path);}
        }
    };

    visit(root, 0, std::make_shared<std::vector<int>>());
    auto finalPath = std::atomic_load(&bestPathWithMax);
    if (finalPath) path = finalPath->path;
    return maxValid.load();
}

size_t findValidMaxTBB_SharedPtr1(std::vector<int>& path,
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
