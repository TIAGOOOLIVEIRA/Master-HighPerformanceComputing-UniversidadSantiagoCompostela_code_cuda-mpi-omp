UNAME_S := $(shell uname -s)

ifeq ($(UNAME_S),Darwin)
    # MacOS
    CC     = mpicc
    CFLAGS = -Xpreprocessor -fopenmp -Ofast -march=native -funroll-loops -ffast-math -Wall -Wextra
    LDFLAGS = -lomp -lm
    PROF_FLAGS = -pg
else
    # Linux HPC
    CC     = mpicc
    CFLAGS = -fopenmp -O3 -march=native -ffast-math -funroll-loops -fopt-info-vec-optimized -Wall -Wextra
    LDFLAGS = -lm
    PROF_FLAGS = -pg
endif

SRC     = pi_integral.c
TARGET  = pi_integral

all: basic

#no optimization to compile with OpenMP and MPI support
#mpicc -fopenmp -O2 -o pi_integral pi_integral.c -lm

#Basic optimized build with SIMD and OpenMP
basic:
	$(CC) $(CFLAGS) $(SRC) -o $(TARGET) $(LDFLAGS)

#Profiling-ready
profile:
	$(CC) $(CFLAGS) $(PROF_FLAGS) $(SRC) -o $(TARGET)_prof $(LDFLAGS)

debug:
	$(CC) -g -fopenmp -Wall -Wextra $(SRC) -o $(TARGET)_debug $(LDFLAGS)

clean:
	rm -f $(TARGET) $(TARGET)_prof $(TARGET)_debug *.o gmon.out

help:
	@echo "Available targets:"
	@echo "  make basic      - Build with OpenMP + SIMD + auto-vectorization"
	@echo "  make profile    - Build with profiling enabled (gprof)"
	@echo "  make debug      - Debug version"
	@echo "  make clean      - Remove binaries"
