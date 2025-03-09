#include <unistd.h>
#include <iostream>
#include <thread>
#include <vector>
#include <chrono>
#include <algorithm>
#include <cmath>

#ifdef ATOMIC_STACK
#include "atomic_concurrent_stack.h"
#elif ATOMIC_FLAG_STACK
#include "atomic_flag_concurrent_stack.h"
#else
#include <atomic>
#include "mutex_concurrent_stack.h"
#endif

using WorkType = size_t;

const int NREPS = 300;
int NThreads = std::thread::hardware_concurrency();
const int NValsPerThread = 1000;
bool Do_print = false;
bool FinalSeqPop = false;
std::vector<float> Times;
int Work = 0;

ConcurrentStack<WorkType> CS;
std::atomic<int> WaitAllReady{0};
std::atomic<int> AllPassed{0};

using myclock_t  = std::chrono::high_resolution_clock;
myclock_t::time_point t0, t1;

template<bool FWD>
WorkType work(WorkType val)
{
  double vin = static_cast<double>(val);

  for (int i = 0; i < Work; i++) {
    if (FWD) {
      vin = 1.01 * vin + 1.;
    } else {
      vin = (vin - 1.) / 1.01;
    }
  }
  
  return static_cast<WorkType>(vin);
}

void thread_func(const int id)
{ std::vector<WorkType> retrieved(NValsPerThread);
  WorkType tmp;
  float measured_time = 0.f;

  for (int nrep = 0; nrep < NREPS; nrep++) {

    while(!CS.empty()) {}

    WaitAllReady++;
    while (WaitAllReady.load() < (nrep + 1) * NThreads) { }

    t0 = myclock_t::now();

    for (int i = 0; i < NValsPerThread; i++) {
      CS.push(work<true>(id * NValsPerThread + i));
    }
    
    if( (nrep < (NREPS - 1)) || !FinalSeqPop ) {
      for (int i = 0; i < NValsPerThread; i++) {
        CS.pop(tmp);
        retrieved[i] = work<false>(tmp);
      }
    }

    t1 = myclock_t::now();
    measured_time += std::chrono::duration<float>(t1 - t0).count();

  }
  
  Times[id] = measured_time;

  AllPassed++;
  while(AllPassed.load() < NThreads) {
    std::this_thread::yield();
  }

  if(!id) {
    WaitAllReady.fetch_sub(NREPS * NThreads);
  }

  while (WaitAllReady.load() != id) { }
  
  if (Do_print) {
    std::cout << "Thread " << id << " time: " << measured_time << '\n';

    if (!FinalSeqPop) {
      for (int val : retrieved) {
        std::cout << val << ' ';
      }
    } else {
      if( id == (NThreads - 1) ) {
        while (!CS.empty()) {
          CS.pop(tmp);
          std::cout << work<false>(tmp) << ' ';
        }
      }
    }

    std::cout << '\n';
  }

  WaitAllReady++;
}

void parse_args(int argc, char ** argv)
{ int c;
  
  while ( -1 != (c = getopt(argc, argv, "hpsw:")) ) {
    switch (c) {
      case 'h':
        std::cout << "-h[elp] -p[rint] -w[work] amount [-s]eq_pop\n";
        exit(EXIT_SUCCESS);
        break;
      case 'p':
        Do_print = true;
        break;
      case 's':
        FinalSeqPop = true;
        break;
      case 'w':
        Work = strtoul(optarg, NULL, 0);
        break;
      default:
        std::cerr << "Unknown option " << (char )c << '\n';
        exit(EXIT_FAILURE);
    }
  }
}

int main(int argc, char ** argv)
{ std::vector<std::thread> v;
  
  if (getenv("OMP_NUM_THREADS")) {
    NThreads = atoi(getenv("OMP_NUM_THREADS"));
  }

  parse_args(argc, argv);
  Times.resize(NThreads);

  std::cout << "Will push and pop " << NValsPerThread << " integers/thread " << NREPS << " times\n";
  std::cout << "Using " << NThreads << " threads and work=" << Work << "\n";

  for (int i = 0; i < NThreads; i++) {
    v.push_back(std::thread(thread_func, i));
  }
  
  for (int i = 0; i < NThreads; i++) {
    v[i].join();
  }
  
  std::cout << "Global time: " << *std::max_element(Times.begin(), Times.end()) << '\n';
  
  return EXIT_SUCCESS;
}
