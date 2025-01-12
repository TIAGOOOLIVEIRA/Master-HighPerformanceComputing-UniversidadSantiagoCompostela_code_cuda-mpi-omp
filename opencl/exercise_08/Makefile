OS := $(shell uname)
OPTIONS:= 

CCFLAGS=-O3 -lm

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



pi_ocl: pi_ocl.c $(COMMON_DIR)/wtime.c $(COMMON_DIR)/device_info.c
	$(CC) $^ $(CCFLAGS) $(OPTIONS) -I $(COMMON_DIR) -o $@

main: pi_ocl.c
	gcc -Wall -g pi_ocl.c -o pi_ocl $(OPTIONS)

clean:
	rm -rf pi_ocl
