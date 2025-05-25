# Detect OS to choose compiler
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S), Darwin)
	CC := gcc-14           # Homebrew GCC on macOS
else
	CC := gcc              # Default GCC on Linux
endif

TARGET ?= jacobi
SRC ?= jacobi.c

MODE ?= basic


ifeq ($(MODE), basic)
	CFLAGS = -O2 -fopenmp -fopt-info-vec -lm
endif

ifeq ($(MODE), report)
	CFLAGS = -O2 -fopenmp -fopt-info-vec -fopt-info-vec-optimized -march=native
endif

ifeq ($(MODE), vectorized)
	CFLAGS = -O3 -fopenmp -march=native -ftree-vectorize -fopt-info-vec
endif

all: $(TARGET)

$(TARGET): $(SRC)
	@echo "Compiling $(SRC) as $(TARGET) with mode: $(MODE)"
	$(CC) $(CFLAGS) -o $(TARGET) $(SRC)

# Clean rule
.PHONY: clean
clean:
	rm -f $(TARGET)
