#include "timer.h"

long diff_nano(struct timespec *start, struct timespec *end)
{
  /* ns */
  return ((end->tv_sec * (1000000000)) + (end->tv_nsec)) -
    ((start->tv_sec * 1000000000) + (start->tv_nsec));
}

long diff_micro(struct timespec *start, struct timespec *end)
{
  /* us */
  return ((end->tv_sec * (1000000)) + (end->tv_nsec / 1000)) -
    ((start->tv_sec * 1000000) + (start->tv_nsec / 1000));
}

long diff_milli(struct timespec *start, struct timespec *end)
{
  /* ms */
  return ((end->tv_sec * 1000) + (end->tv_nsec / 1000000)) -
    ((start->tv_sec * 1000) + (start->tv_nsec / 1000000));
}

long diff_seconds(struct timespec *start, struct timespec *end)
{
  /* s */
  return (end->tv_sec + (end->tv_nsec / 1000000000)) -
    (start->tv_sec + (start->tv_nsec / 1000000000));
}
