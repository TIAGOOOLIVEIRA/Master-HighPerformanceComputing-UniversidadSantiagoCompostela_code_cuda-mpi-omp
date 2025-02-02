#include <unistd.h>
#include <cstdlib>
#include <cassert>
#include <iostream>
#include <chrono>
#include <thread>
#include <mutex>
#include <condition_variable>

struct buffer_t {
  
  static constexpr size_t BSIZE = 1024;
  
  size_t buf[BSIZE];
  int occupied = 0;
  int nextin = 0;
  int nextout = 0;
  std::mutex mutex;
  std::condition_variable more;
  std::condition_variable less;
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

  std::unique_lock<std::mutex> my_lock(b.mutex);
   
  while (b.occupied >= buffer_t::BSIZE) {
    block = 1;
    b.less.wait(my_lock);
  }
  
  assert(b.occupied < buffer_t::BSIZE);

  b.buf[b.nextin++] = item;

  b.nextin %= buffer_t::BSIZE;
  b.occupied++;

  my_lock.unlock();
  
  producer_blocked += block;

    /* now: either b.occupied < BSIZE and b.nextin is the index
       of the next empty slot in the buffer, or
       b.occupied == BSIZE and b.nextin is the index of the
       next (occupied) slot that will be emptied by a consumer
       (such as b.nextin == b.nextout) */

   b.more.notify_one();
}

size_t consumer(buffer_t& b)
{ size_t item;
  size_t block = 0;

  std::unique_lock<std::mutex> my_lock(b.mutex);
 
  while(b.occupied <= 0) {
    block = 1;
    b.more.wait(my_lock);
  }
  
  assert(b.occupied > 0);

  item = b.buf[b.nextout++];
  b.nextout %= buffer_t::BSIZE;
  b.occupied--;

  my_lock.unlock();

  consumer_blocked += block;

    /* now: either b.occupied > 0 and b.nextout is the index
       of the next occupied slot in the buffer, or
       b.occupied == 0 and b.nextout is the index of the next
       (empty) slot that will be filled by a producer (such as
       b.nextout == b.nextin) */

  b.less.notify_one();

  return item;
}

#include "prod_cons_common.cpp"
