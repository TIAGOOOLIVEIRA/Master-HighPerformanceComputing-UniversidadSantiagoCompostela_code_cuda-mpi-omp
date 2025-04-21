CUDACXX=nvcc
CUDACXXFLAGS=-arch=sm_70 -O3
CXXFLAGS=-march=native -fopenmp
NSYS=nsys profile
NSYSFLAGS=--stats=true --force-overwrite=true

all: mergesort
mergesort: mergesort.cu
	$(CUDACXX) $(CUDACXXFLAGS) -Xcompiler="$(CXXFLAGS)" mergesort.cu -o mergesort

profile: mergesort
	$(NSYS) $(NSYSFLAGS) -o mergesort-report ./mergesort

clean:
	rm -f mergesort *.qdrep *.sqlite