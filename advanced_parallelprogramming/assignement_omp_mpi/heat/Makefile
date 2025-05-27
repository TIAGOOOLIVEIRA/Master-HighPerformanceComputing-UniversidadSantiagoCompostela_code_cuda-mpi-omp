# Detect OS to choose compiler
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S), Darwin)
	CC := gcc-14           # Homebrew GCC on macOS
else
	CC := gcc              # Default GCC on Linux
endif

TARGET ?= heat
SRC ?= heat.c

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

$(TARGET): $(SRC)
	@echo "Compiling $(SRC) as $(TARGET) with mode: $(MODE)"
	$(CC) $(CFLAGS) -o $(TARGET) $(SRC) $(LDFLAGS)

#Run the application to generate gmon.out
.PHONY: run-profile
run-profile: $(TARGET)
	@echo "Running $(TARGET) to generate gmon.out..."
	./$(TARGET)

#Generate GProf report
.PHONY: gprof-report
gprof-report: gmon.out
	@echo "Generating gprof report..."
	gprof $(TARGET) gmon.out > gprof-report.txt
	@echo "Report saved to gprof-report.txt"

.PHONY: clean
clean:
	rm -f $(TARGET) gmon.out gprof-report.txt