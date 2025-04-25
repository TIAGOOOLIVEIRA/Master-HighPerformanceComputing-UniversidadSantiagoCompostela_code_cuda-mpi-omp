CUDACXX=nvcc
CUDACXXFLAGS=-arch=sm_70 -O3 -lineinfo -Xptxas -v -Xcudafe "--display_error_number"
CXXFLAGS=-march=native -fopenmp
NSYS=nsys profile
NSYSFLAGS=--stats=true --force-overwrite=true
VTUNE=vtune
VTUNEFLAGS=-collect gpu-hotspots
VTUNE_RESULT_DIR=vtune-report-gpu
NCU=ncu
NCUFLAGS=--target-processes all --launch-count 1 -f --kernel-name "sort_kernel" --set memory-workload-analysis

all: mergesort

mergesort: mergesort.cu
	$(CUDACXX) $(CUDACXXFLAGS) -Xcompiler="$(CXXFLAGS)" mergesort.cu -o mergesort

profile: mergesort
	$(NSYS) $(NSYSFLAGS) -o mergesort-report ./mergesort

vtune-cpu: mergesort
	$(VTUNE) -collect hotspots -result-dir vtune-report-cpu ./mergesort

vtune-gpu: mergesort
	@if command -v $(VTUNE) >/dev/null 2>&1; then \
		echo "Running VTune GPU profiling..."; \
		$(VTUNE) $(VTUNEFLAGS) -result-dir $(VTUNE_RESULT_DIR) ./mergesort; \
	else \
		echo "Warning: VTune is not installed or not in PATH. Skipping GPU profiling."; \
	fi

ncu: mergesort
	$(NCU) $(NCUFLAGS) -o mergesort_ncu_t4 ./mergesort

clean:
	rm -f mergesort *.qdrep *.sqlite
	rm -rf vtune-report* mergesort-report* mergesort_ncu_t4.ncu-rep
