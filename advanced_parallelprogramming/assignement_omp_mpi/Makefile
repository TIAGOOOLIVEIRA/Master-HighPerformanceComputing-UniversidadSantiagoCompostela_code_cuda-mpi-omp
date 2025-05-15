# Makefile for compiling OpenMP application on macOS and Linux

# Detect platform
UNAME_S := $(shell uname -s)

# Default target
TARGET = multf

# Source file
SRC = multf.c

# Compiler flags
CFLAGS = -fopenmp -O3 -march=native

# Platform-specific settings
ifeq ($(UNAME_S), Darwin)
    # macOS (assumes Homebrew GCC installed)
    CC = gcc-14
else
    # Linux (uses default gcc)
    CC = gcc
endif

# Build target
$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $(TARGET) $(SRC)

# Clean up
clean:
	rm -f $(TARGET)
