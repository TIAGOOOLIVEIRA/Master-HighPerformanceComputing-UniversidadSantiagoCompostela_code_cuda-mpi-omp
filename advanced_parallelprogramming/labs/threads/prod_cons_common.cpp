
#include <vector>
#include <functional>

int NHWThreads = 0;

void collide_computation(const int id, int& res)
{
  res = rec_fib(46) - id;
}

void producer_f()
{
  for (size_t i = 1; i < NValues; i++) {
    produce(buffer, i);
  }

  produce(buffer, 0); // means exit
}

void consumer_f()
{ size_t val;
  size_t tot = 0;

  do {
    val = consumer(buffer);
    tot += val;
    const size_t r = Do_fib ? rec_fib(val % Fib_mod) : 0;
    if (Do_print) {
      std::cout << val << " fib(%32)=" << r << '\n';
    }
  } while(val);
  std::cout << "Total values received=" << tot << '\n';
}

void parse_args(int argc, char ** argv)
{ int c;

  while ( -1 != (c = getopt(argc, argv, "hfm:p")) ) {
    switch (c) {
      case 'h':
        std::cout << "-h[elp] -f[ib] -p[rint] -m[odulus_fib]\n";
        exit(EXIT_SUCCESS);
        break;
      case 'f':
        Do_fib = true;
        break;
      case 'm':
        Fib_mod = (size_t) strtoul(optarg, NULL, 0);
        break;
      case 'p':
        Do_print = true;
        break;
      default:
        std::cerr << "Unknown option " << (char )c << '\n';
        exit(EXIT_FAILURE);
    }
  }
  
  std::cout << "f[ib]         =" << (Do_fib ? 'Y' : 'N')   << '\n';
  std::cout << "m[odulus_fib] =" << Fib_mod << '\n';
  std::cout << "p[rint]       =" << (Do_print ? 'Y' : 'N') << '\n';
  std::cout << "Using " << NHWThreads << " collision threads\n";
}

int main(int argc, char ** argv)
{ int Global_result_collide = 0;
  std::vector<std::thread> Collide_computations;
  
  if (getenv("OMP_NUM_THREADS")) {
    NHWThreads = atoi(getenv("OMP_NUM_THREADS"));
    if (NHWThreads < 2) {
      std::cerr << NHWThreads << "< 2 threads. Using 2\n";
      NHWThreads = 0;
    } else {
      NHWThreads -= 2;
    }
  }

  parse_args(argc, argv);
  std::vector<int> Results_collide(NHWThreads);

  for (int i = 0; i < NHWThreads; i++) {
    Collide_computations.push_back(std::thread(collide_computation, i, std::ref(Results_collide[i])));
  }

  auto t0 = std::chrono::high_resolution_clock::now();

  std::thread thread1(producer_f);

  consumer_f();

  thread1.join();

  for (int i = 0; i < NHWThreads; i++) {
    Collide_computations[i].join();
    Global_result_collide += Results_collide[i];
  }


  auto t1 = std::chrono::high_resolution_clock::now();
  double time = std::chrono::duration<double>(t1 - t0).count();

  if (NHWThreads) {
    std::cout << "Collide computation result=" << Global_result_collide << "\n";
  }
  std::cout << "Producer was blocked " << producer_blocked << " times\n";
  std::cout << "Consumer was blocked " << consumer_blocked << " times\n";
  std::cout << "Runtime              " << time << " s.\n";

  return 0;
}
