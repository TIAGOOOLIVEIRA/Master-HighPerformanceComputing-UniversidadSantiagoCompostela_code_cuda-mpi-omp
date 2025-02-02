#include <thread>
#include <random>
#include <vector>
#include <iostream>
#include <shared_mutex>
#include <string>
#include <map>
#include <chrono>
#include <algorithm>

const int NOps = 1000000;

const std::vector<std::string> Names = {"Adriana", "Fernando", "Francisco", "Gonzalo", "Luis", "Carlos", "Basilio", "Diego", "Jose", "Millan", "Uxia", "Dario", "Michel", "Victor", "Ismael", "Maria", "Juan", "Ramon", "Patricia", "Margarita", "Jorge", "Nuria", "Rosa", "Yolanda", "Cristina", "Rodrigo", "Tomas", "Elvira", "Pedro", "Bernardo"};

enum OpType { Read = 0, Add, Erase};

struct Op {
  OpType kind;
  int name_index; // an index into vector Names
  size_t number;
};

int NThreads = 1;
std::vector<std::thread> Threads;
std::vector<Op> OpsContainer;
std::map<std::string, size_t> PhoneBook;
std::vector<std::vector<size_t>> Results;

void basic_init()
{ size_t i = 0;

  PhoneBook.clear();
  for (const auto& name : Names) {
    PhoneBook[name] = ++i;
  }
}

void mutex_f(const int id)
{ static std::mutex Global_mutex;

  std::vector<size_t>& results = Results[id];
  const int limit = (id + 1) * NOps;
  
  for (int i = id * NOps; i < limit; i++) {
    const Op& curr_op = OpsContainer[i];
    std::lock_guard<std::mutex> lock(Global_mutex);
    const auto it = PhoneBook.find(Names[curr_op.name_index]);
    switch (curr_op.kind) {
      case Read:
        if (it != PhoneBook.end()) {
          results.push_back(it->second);
        }
        break;
      case Add:
        if (it != PhoneBook.end()) {
          it->second = curr_op.number;
        } else {
          PhoneBook[Names[curr_op.name_index]] = curr_op.number;
        }
        break;
      case Erase:
        if (it != PhoneBook.end()) {
          PhoneBook.erase(it);
        }
        break;
      default:
        break;
    }
  }
}

void mutex_s(const int id)
{ static std::shared_timed_mutex Global_mutex;
  
  std::vector<size_t>& results = Results[id];
  const int limit = (id + 1) * NOps;

  for (int i = id * NOps; i < limit; i++) {
    const Op& curr_op = OpsContainer[i];
    if (curr_op.kind == Read) {
      std::shared_lock<std::shared_timed_mutex> readLock(Global_mutex);
      const auto it = PhoneBook.find(Names[curr_op.name_index]);
      if (it != PhoneBook.end()) {
        results.push_back(it->second);
      }
    } else {
      std::lock_guard<std::shared_timed_mutex> lock(Global_mutex);
      const auto it = PhoneBook.find(Names[curr_op.name_index]);
      if (curr_op.kind == Erase) {
        if (it != PhoneBook.end()) {
          PhoneBook.erase(it);
        }
      } else { // Add
        if (it != PhoneBook.end()) {
          it->second = curr_op.number;
        } else {
          PhoneBook[Names[curr_op.name_index]] = curr_op.number;
        }
      }
    }
  }

}

int main()
{ size_t tmp;

  if (getenv("OMP_NUM_THREADS")) {
    NThreads = atoi(getenv("OMP_NUM_THREADS"));
  }
  Results.resize(NThreads);
  OpsContainer.reserve(NOps * NThreads);

  std::cout << "Performing " << NOps << " operations in each one of the " << NThreads << " threads\n";

  std::mt19937 gen(981);
  std::uniform_int_distribution<size_t> dist(0);
  
  for (int i = 0; i < NOps * NThreads; i++) {
    size_t tmp = dist(gen) % 100;
    const OpType kind = static_cast<OpType>((tmp < 98) ? 0 : (100 - tmp)); // 98% reads 1% writes 1% erases
    const int name_index = dist(gen) % Names.size();
    OpsContainer.push_back({kind, name_index, dist(gen) % 999999999 + 1});
  }
  
  basic_init();
  auto t0 = std::chrono::high_resolution_clock::now();
  
  for (int i = 0; i < NThreads; i++) {
    Threads.push_back(std::thread(mutex_f, i));
  }

  for (int i = 0; i < NThreads; i++) {
    Threads[i].join();
  }
  
  auto t1 = std::chrono::high_resolution_clock::now();
  std::cout << "       mutex Runtime:" << std::chrono::duration<double>(t1 - t0).count() << " s.\n";
  
  tmp = 0;
  for (auto& v : Results) {
    tmp += v.size();
    v.clear();
  }
  std::cout << tmp << " phone numbers retrieved\n";
  
  Threads.clear();
  
  ////
  
  basic_init();
  t0 = std::chrono::high_resolution_clock::now();
  
  for (int i = 0; i < NThreads; i++) {
    Threads.push_back(std::thread(mutex_s, i));
  }
  
  for (int i = 0; i < NThreads; i++) {
    Threads[i].join();
  }
  
  t1 = std::chrono::high_resolution_clock::now();
  std::cout << "shared_mutex Runtime:" << std::chrono::duration<double>(t1 - t0).count() << " s.\n";
  
  tmp = 0;
  for (auto& v : Results) {
    tmp += v.size();
    v.clear();
  }
  std::cout << tmp << " phone numbers retrieved\n";
}
