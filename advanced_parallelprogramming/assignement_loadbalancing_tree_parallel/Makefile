# Makefile for compiling ppa_project_par.cpp
# Supports macOS with Homebrew and generic HPC environments

# -----------------------------
# Configuration Section
# -----------------------------

# Compiler and flags
CXX        = clang++
CXXFLAGS   = -std=c++17 -O3

# Target binary
TARGET     = ppa_project_par
SRC        = ppa_project_par.cpp

# -----------------------------
# macOS/Homebrew paths (adjust if needed)
# -----------------------------
MAC_OPENMP_INC = /opt/homebrew/opt/libomp/include
MAC_OPENMP_LIB = /opt/homebrew/opt/libomp/lib
MAC_TBB_INC    = /opt/homebrew/include
MAC_TBB_LIB    = /opt/homebrew/lib

# -----------------------------
# Flags for macOS build
# -----------------------------
MACFLAGS   = -Xpreprocessor -fopenmp \
             -I$(MAC_OPENMP_INC) -L$(MAC_OPENMP_LIB) -lomp \
             -I$(MAC_TBB_INC) -L$(MAC_TBB_LIB) -ltbb

# -----------------------------
# Default build target (macOS)
# -----------------------------
all: export OMP_NUM_THREADS=8
all: export TBB_NUM_THREADS=8
all:
	$(CXX) $(CXXFLAGS) $(SRC) -o $(TARGET) $(MACFLAGS)
	@echo "✅ Build complete: $(TARGET)"

# -----------------------------
# Clean target
# -----------------------------
clean:
	rm -f $(TARGET)
	@echo "🧹 Cleaned build artifacts"

# -----------------------------
# HPC build target (example)
# Customize include/lib paths or use `module load`
# -----------------------------
hpc:
	$(CXX) $(CXXFLAGS) $(SRC) -fopenmp -ltbb -o $(TARGET)
	@echo "✅ HPC Build complete: $(TARGET)"
