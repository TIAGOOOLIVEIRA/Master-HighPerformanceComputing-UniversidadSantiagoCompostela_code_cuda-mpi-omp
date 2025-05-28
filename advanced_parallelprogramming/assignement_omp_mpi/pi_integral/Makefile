UNAME_S := $(shell uname -s)

# Platform-specific compiler and flags
ifeq ($(UNAME_S),Darwin)
    CC         = mpicc
    BASEFLAGS  = -Xpreprocessor -fopenmp -Wall -Wextra
    LDFLAGS    = -lomp -lm
else
    CC         = mpicc
    BASEFLAGS  = -fopenmp -Wall -Wextra
    LDFLAGS    = -lm
endif

CFLAGS_BASIC      = $(BASEFLAGS) -O3 -march=native -ffast-math -fopt-info-vec
CFLAGS_PROFILE    = $(CFLAGS_BASIC) -pg
CFLAGS_DEBUG      = $(BASEFLAGS) -g
CFLAGS_VTUNE      = $(CFLAGS_BASIC) -g -fno-inline
CFLAGS_AGGRESSIVE = $(BASEFLAGS) -O3 -march=native -funroll-loops -ffast-math -finline-functions -ftree-vectorize -fopt-info-vec-optimized

SRC        = pi_integral.c
TARGET     = pi_integral
SHAREDDIR  = ../shared
SHAREDOBJ  = place_report_mpi.o
SHAREDSRC  = $(SHAREDDIR)/place_report_mpi.c

all: basic

basic: $(SHAREDOBJ)
	$(CC) $(CFLAGS_BASIC) $(SRC) $(SHAREDOBJ) -o $(TARGET) $(LDFLAGS)

profile: $(SHAREDOBJ)
	$(CC) $(CFLAGS_PROFILE) $(SRC) $(SHAREDOBJ) -o $(TARGET)_prof $(LDFLAGS)

vtune: $(SHAREDOBJ)
	$(CC) $(CFLAGS_VTUNE) $(SRC) $(SHAREDOBJ) -o $(TARGET)_vtune $(LDFLAGS)

debug: $(SHAREDOBJ)
	$(CC) $(CFLAGS_DEBUG) $(SRC) $(SHAREDOBJ) -o $(TARGET)_debug $(LDFLAGS)

aggressive: $(SHAREDOBJ)
	$(CC) $(CFLAGS_AGGRESSIVE) $(SRC) $(SHAREDOBJ) -o $(TARGET)_opt $(LDFLAGS)

$(SHAREDOBJ): $(SHAREDSRC)
	$(CC) -c $(BASEFLAGS) -O2 $< -o $@

clean:
	rm -f $(TARGET) $(TARGET)_prof $(TARGET)_debug $(TARGET)_vtune $(TARGET)_opt *.o gmon.out

help:
	@echo "Available targets:"
	@echo "  make basic       - Build with OpenMP + SIMD + base optimization"
	@echo "  make aggressive  - Aggressive optimization with vectorization"
	@echo "  make profile     - Build with gprof profiling enabled"
	@echo "  make vtune       - Build for VTune analysis (with debug symbols)"
	@echo "  make debug       - Debug version"
	@echo "  make clean       - Remove binaries and objects"
