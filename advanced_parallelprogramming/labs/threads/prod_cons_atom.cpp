#include <unistd.h>
#include <cstdlib>
#include <cassert>
#include <iostream>
#include <chrono>
#include <thread>
#include <atomic>

struct buffer_t {
  
  static constexpr size_t BSIZE = 1024;
  
  size_t buf[BSIZE];
  std::atomic<int> occupied {0};
  int nextin = 0;
  int nextout = 0;
};

constexpr int NValues = 65536;

buffer_t buffer;
size_t producer_blocked = 0;
size_t consumer_blocked = 0;
bool Do_print = false;
bool Do_fib = false;
size_t Fib_mod = 30;

size_t rec_fib(size_t n) {
  if (n < 2) {
    return n;
  } else {
    return rec_fib(n - 1) + rec_fib(n - 2);
  }
}

void produce(buffer_t& b, size_t item)
{ size_t block = 0;
  
  while (b.occupied.load() >= buffer_t::BSIZE) {
    block = 1;
  }
  
  assert(b.occupied.load() < buffer_t::BSIZE);

  b.buf[b.nextin++] = item;

  b.nextin %= buffer_t::BSIZE;
  b.occupied++;
  
  producer_blocked += block;

    /* now: either b.occupied < BSIZE and b.nextin is the index
       of the next empty slot in the buffer, or
       b.occupied == BSIZE and b.nextin is the index of the
       next (occupied) slot that will be emptied by a consumer
       (such as b.nextin == b.nextout) */
}

size_t consumer(buffer_t& b)
{ size_t item;
  size_t block = 0;
 
  while(b.occupied.load() <= 0) {
    block = 1;
  }
  
  assert(b.occupied.load() > 0);

  item = b.buf[b.nextout++];
  b.nextout %= buffer_t::BSIZE;
  b.occupied--;

  consumer_blocked += block;

    /* now: either b.occupied > 0 and b.nextout is the index
       of the next occupied slot in the buffer, or
       b.occupied == 0 and b.nextout is the index of the next
       (empty) slot that will be filled by a producer (such as
       b.nextout == b.nextin) */

  return item;
}

#include "prod_cons_common.cpp"
