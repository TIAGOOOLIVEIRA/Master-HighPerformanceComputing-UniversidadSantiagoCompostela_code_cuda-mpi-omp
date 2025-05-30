# Detect OS to choose compiler
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S), Darwin)
	CC := gcc-14           # Homebrew GCC on macOS
else
	CC := gcc              # Default GCC on Linux
endif

TARGET ?= jacobi
SRC ?= jacobi.c
SHARED_DIR = ../shared
OBJS = place_report_omp.o

MODE ?= basic

ifeq ($(MODE), basic)
	CFLAGS = -O2 -fopenmp -fopt-info-vec
endif

ifeq ($(MODE), report)
	CFLAGS = -O2 -fopenmp -fopt-info-vec -fopt-info-vec-optimized -march=native
endif

ifeq ($(MODE), vectorized)
	CFLAGS = -O3 -fopenmp -march=native -ftree-vectorize -fopt-info-vec
endif

ifeq ($(MODE), profile)
	CFLAGS = -pg -O2 -fopenmp
endif

LDFLAGS = -lm
ifeq ($(MODE), profile)
	LDFLAGS += -pg
endif

all: $(TARGET)

$(TARGET): $(SRC) $(OBJS)
	@echo "Compiling $(SRC) as $(TARGET) with mode: $(MODE)"
	$(CC) $(CFLAGS) -o $(TARGET) $(SRC) $(OBJS) $(LDFLAGS)

# Build object from shared source
place_report_omp.o: $(SHARED_DIR)/place_report_omp.c $(SHARED_DIR)/place_report_omp.h
	$(CC) $(CFLAGS) -c $(SHARED_DIR)/place_report_omp.c -o place_report_omp.o

# Run the application to generate gmon.out (only meaningful in 'profile' mode)
.PHONY: run-profile
run-profile: $(TARGET)
	@echo "Running $(TARGET) to generate gmon.out..."
	./$(TARGET)

# Generate GProf report
.PHONY: gprof-report
gprof-report: gmon.out
	@echo "Generating gprof report..."
	gprof $(TARGET) gmon.out > gprof-report.txt
	@echo "Report saved to gprof-report.txt"

# Clean rule
.PHONY: clean
clean:
	rm -f $(TARGET) *.o gmon.out gprof-report.txt
