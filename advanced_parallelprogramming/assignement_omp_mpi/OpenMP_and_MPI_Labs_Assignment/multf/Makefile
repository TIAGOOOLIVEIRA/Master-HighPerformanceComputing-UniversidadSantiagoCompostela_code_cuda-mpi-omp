# Auto-detect platform
UNAME_S := $(shell uname -s)

ifeq ($(UNAME_S), Darwin)
    CC := gcc-14
else
    CC := gcc
endif

# Default target and source
TARGET ?= multf
SRC ?= $(TARGET).c

# Build mode: sequential, parallel, simd, vector
MODE ?= parallel

# Output binary
OUT ?= $(TARGET)

# Compiler flags by build mode
ifeq ($(MODE), sequential)
    CFLAGS = -O2 -fno-tree-vectorize -fopenmp
endif

ifeq ($(MODE), parallel)
    CFLAGS = -O2 -fopenmp -fopt-info-vec -march=native
endif

ifeq ($(MODE), simd)
    CFLAGS = -O2 -fopenmp -fopt-info-vec -march=native
endif

ifeq ($(MODE), vector)
    CFLAGS = -O3 -fopenmp -ftree-vectorize -fopt-info-vec -march=native
endif

# Build rule
$(OUT): $(SRC)
	@echo "Compiling $(SRC) with mode: $(MODE)"
	$(CC) $(CFLAGS) -o $(OUT) $(SRC)

.PHONY: clean
clean:
	@echo "Cleaning build artifacts..."
	rm -f multf multf_vec
