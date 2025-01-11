OS := $(shell uname)
OPTIONS:= 

CCFLAGS=-O3 -lm -fopenmp

COMMON_DIR = ../C_common

CCFLAGS += -D DEVICE=$(DEVICE)

ifeq ($(OS),Darwin)
	OPTIONS += -framework OpenCL
else
	OPTIONS += -l OpenCL
endif

ifndef CC
	CC = gcc
endif



oclHeterogDevicesProc: oclHeterogDevicesProc.c $(COMMON_DIR)/wtime.c $(COMMON_DIR)/device_info.c
	$(CC) $^ $(CCFLAGS) $(OPTIONS) -I $(COMMON_DIR) -o $@

main: oclHeterogDevicesProc.c
	gcc -Wall -g oclHeterogDevicesProc.c -o oclHeterogDevicesProc $(OPTIONS)  -fopenmp

clean:
	rm -rf oclHeterogDevicesProc