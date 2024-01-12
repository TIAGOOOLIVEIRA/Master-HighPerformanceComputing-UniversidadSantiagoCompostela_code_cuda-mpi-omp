OS := $(shell uname)
OPTIONS:= 

ifeq ($(OS),Darwin)
	OPTIONS += -framework OpenCL
else
	OPTIONS += -l OpenCL
endif

main: main.c
	gcc -Wall -g oclHeterogDevicesProc.c -o oclHeterogDevicesProc $(OPTIONS)

clean:
	rm -rf oclHeterogDevicesProc