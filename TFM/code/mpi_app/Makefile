# Makefile for HPC Metagenomic Clustering Application
# File: Makefile
# Location: /opt/hpc-clustering/Makefile

# Compilers
CC = mpicc
CXX = mpicxx
NVCC = nvcc

# Compiler flags
CFLAGS = -O3 -Wall -std=c99 -fopenmp -march=native
CXXFLAGS = -O3 -Wall -std=c++17 -fopenmp -march=native
NVCCFLAGS = -O3 -arch=sm_70 -std=c++17

# Include directories
INCLUDES = -I/opt/arrow/include \
           -I/opt/parquet/include \
           -I$(CUDA_HOME)/include \
           -I/usr/include/openmpi \
           -I/usr/local/include

# Library directories and libraries
LIBDIRS = -L/opt/arrow/lib \
          -L/opt/parquet/lib \
          -L$(CUDA_HOME)/lib64 \
          -L/usr/lib/x86_64-linux-gnu/openmpi/lib \
          -L/usr/local/lib

# Core libraries (always needed)
LIBS = -larrow -lparquet -lgomp -lm -lstdc++

# CUDA libraries (only when USE_CUDA is defined)
CUDA_LIBS = -lcuda -lcudart -lcublas

# Source files
MAIN_SOURCE = hpc_metagenome_cluster.c
TARGET = hpc_metagenome_cluster
TARGET_GPU = hpc_metagenome_cluster_gpu

# CUDA source files (optional)
CUDA_SOURCES = cuda_kernels.cu
CUDA_OBJECTS = $(CUDA_SOURCES:.cu=.o)

# Object files
OBJECTS = $(MAIN_SOURCE:.c=.o)

# Default target - CPU version
all: $(TARGET)

# GPU-enabled build target
cuda: CFLAGS += -DUSE_CUDA
cuda: CXXFLAGS += -DUSE_CUDA
cuda: LIBS += $(CUDA_LIBS)
cuda: $(TARGET_GPU)

# CPU-only build
build-cpu: clean $(TARGET)
	@echo "CPU version built successfully"

# GPU build  
build-gpu: clean cuda
	@echo "GPU version built successfully"

# Link CPU version
$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIBDIRS) $(LIBS)
	@echo "Linking complete: $(TARGET)"

# Link GPU version
$(TARGET_GPU): CFLAGS += -DUSE_CUDA
$(TARGET_GPU): $(OBJECTS) $(CUDA_OBJECTS)
	$(CXX) $(CXXFLAGS) -DUSE_CUDA -o $@ $^ $(LIBDIRS) $(LIBS) $(CUDA_LIBS)
	@echo "Linking complete: $(TARGET_GPU)"

# Compile C source files
%.o: %.c
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@
	@echo "Compiled: $<"

# Compile CUDA source files
%.o: %.cu
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@
	@echo "Compiled CUDA: $<"

# Clean build artifacts
clean:
	rm -f $(OBJECTS) $(CUDA_OBJECTS) $(TARGET) $(TARGET_GPU)
	@echo "Cleaned build artifacts"

# Install dependencies (run as root on Ubuntu 20.04)
install-deps:
	@echo "Installing dependencies..."
	apt-get update
	apt-get install -y build-essential cmake pkg-config
	apt-get install -y libopenmpi-dev openmpi-bin openmpi-common
	# Install Arrow/Parquet from official packages
	apt-get install -y -V ca-certificates lsb-release wget
	wget https://apache.jfrog.io/artifactory/arrow/$(lsb_release --id --short | tr 'A-Z' 'a-z')/apache-arrow-apt-source-latest-$(lsb_release --codename --short).deb
	dpkg -i apache-arrow-apt-source-latest-$(lsb_release --codename --short).deb
	apt-get update
	apt-get install -y libarrow-dev libparquet-dev libarrow-python-dev
	@echo "Core dependencies installed"

# Install CUDA toolkit (if NVIDIA GPU available)
install-cuda:
	@echo "Installing CUDA toolkit..."
	wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
	dpkg -i cuda-keyring_1.0-1_all.deb
	apt-get update
	apt-get install -y cuda-toolkit-11-8
	@echo "CUDA installed"

# Test build with small dataset
test: $(TARGET)
	@echo "Running test build..."
	mkdir -p /tmp/test_clustering
	echo "sample_id,feature_0,feature_1,feature_2" > /tmp/test_clustering/test_embeddings_part_0000.csv
	echo "seq001,0.1,0.2,0.3" >> /tmp/test_clustering/test_embeddings_part_0000.csv
	echo "seq002,0.4,0.5,0.6" >> /tmp/test_clustering/test_embeddings_part_0000.csv
	mpirun -np 1 ./$(TARGET) /tmp/test_clustering /tmp/test_output 2 10
	@echo "Test completed - check /tmp/test_output"

# Performance profiling build
profile: CFLAGS += -pg -g
profile: CXXFLAGS += -pg -g
profile: $(TARGET)
	@echo "Profiling build complete"

# Debug build
debug: CFLAGS += -g -DDEBUG -O0
debug: CXXFLAGS += -g -DDEBUG -O0
debug: $(TARGET)
	@echo "Debug build complete"

# Optimized build for specific CPU architecture
optimize-native: CFLAGS += -march=native -mtune=native -flto
optimize-native: CXXFLAGS += -march=native -mtune=native -flto
optimize-native: $(TARGET)
	@echo "Native optimized build complete"

# Check for required libraries
check-libs:
	@echo "Checking for required libraries..."
	@pkg-config --exists arrow || echo "WARNING: Arrow not found"
	@pkg-config --exists parquet || echo "WARNING: Parquet not found"
	@which mpicc >/dev/null || echo "WARNING: MPI compiler not found"
	@echo "Library check complete"

# Show build configuration
show-config:
	@echo "Build Configuration:"
	@echo "  CC: $(CC)"
	@echo "  CXX: $(CXX)"
	@echo "  CFLAGS: $(CFLAGS)"
	@echo "  CXXFLAGS: $(CXXFLAGS)"
	@echo "  INCLUDES: $(INCLUDES)"
	@echo "  LIBS: $(LIBS)"
	@echo "  TARGET: $(TARGET)"

# Create deployment package
package: clean build-cpu
	@echo "Creating deployment package..."
	mkdir -p hpc-clustering-package
	cp $(TARGET) hpc-clustering-package/
	cp $(MAIN_SOURCE) hpc-clustering-package/
	cp Makefile hpc-clustering-package/
	tar -czf hpc-clustering-package.tar.gz hpc-clustering-package/
	rm -rf hpc-clustering-package/
	@echo "Package created: hpc-clustering-package.tar.gz"

# Install to system location
install: $(TARGET)
	@echo "Installing to /usr/local/bin..."
	cp $(TARGET) /usr/local/bin/
	chmod +x /usr/local/bin/$(TARGET)
	@echo "Installation complete"

# Uninstall from system
uninstall:
	rm -f /usr/local/bin/$(TARGET)
	@echo "Uninstalled"

.PHONY: all clean install-deps install-cuda test profile debug optimize-native check-libs show-config package install uninstall build-cpu build-gpu cuda