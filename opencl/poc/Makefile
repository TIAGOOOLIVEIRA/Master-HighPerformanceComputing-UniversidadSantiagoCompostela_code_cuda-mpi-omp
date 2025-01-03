# Compiler and flags
CC = gcc
CFLAGS = -Wall -g

# Detect the operating system
OS := $(shell uname)

# OpenCL library options based on OS
ifeq ($(OS), Darwin)  # For macOS
    LDFLAGS = -framework OpenCL
else                  # For Linux and others
    LDFLAGS = -lOpenCL
endif

# Target binary
TARGET = clDetectCPU_GPU

# Build rule
$(TARGET): clDetectCPU_GPU.c
	$(CC) $(CFLAGS) clDetectCPU_GPU.c -o $(TARGET) $(LDFLAGS)

# Clean rule
clean:
	rm -f $(TARGET)
