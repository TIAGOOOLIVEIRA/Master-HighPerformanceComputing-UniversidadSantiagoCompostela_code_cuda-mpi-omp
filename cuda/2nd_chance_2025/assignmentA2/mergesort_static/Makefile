CUDACXX=nvcc
CUDACXXFLAGS=-O3 -lineinfo -Xptxas -v -Xcudafe "--display_error_number"
CXXFLAGS=-march=native -fopenmp

NSYS=nsys profile
NSYSFLAGS=--stats=true --force-overwrite=true

VTUNE=vtune
VTUNEFLAGS_CPU=-collect hotspots
VTUNEFLAGS_GPU=-collect gpu-hotspots
VTUNE_RESULT_DIR_CPU=vtune-report-cpu
VTUNE_RESULT_DIR_GPU=vtune-report-gpu

NCU=ncu
NCUFLAGS=--target-processes all --launch-skip 0 --launch-count 1 --kernel-name "sort_kernel_phase1" --set full
NCU_OUTPUT=mergesort_ncu_report

#Detect GPU arch dynamically (if possible), else fallback to compute_75/sm_75
GPU_ARCH := $(shell if command -v nvidia-smi >/dev/null 2>&1; then \
	nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1 | sed 's/\.//'; \
	else echo 75; fi)

GENCODE_FLAGS := -gencode arch=compute_$(GPU_ARCH),code=sm_$(GPU_ARCH)

all: mergesort

mergesort: mergesort.cu
	$(CUDACXX) $(GENCODE_FLAGS) $(CUDACXXFLAGS) -Xcompiler="$(CXXFLAGS)" mergesort.cu -o mergesort

profile: mergesort
	$(NSYS) $(NSYSFLAGS) -o mergesort-report ./mergesort

vtune-cpu: mergesort
	rm -rf $(VTUNE_RESULT_DIR_CPU)
	$(VTUNE) $(VTUNEFLAGS_CPU) -result-dir $(VTUNE_RESULT_DIR_CPU) ./mergesort
	$(VTUNE) -collect performance-snapshot -collect memory-access -collect hotspots -collect threading -- ./mergesort

vtune-gpu: mergesort
	rm -rf $(VTUNE_RESULT_DIR_GPU)
	@if command -v $(VTUNE) >/dev/null 2>&1; then \
		echo "Running VTune GPU profiling"; \
		$(VTUNE) $(VTUNEFLAGS_GPU) -result-dir $(VTUNE_RESULT_DIR_GPU) ./mergesort; \
	else \
		echo "Warning: VTune is not installed or not in PATH. Skipping GPU profiling."; \
	fi

ncu: mergesort
	$(NCU) $(NCUFLAGS) -o $(NCU_OUTPUT) ./mergesort

clean:
	rm -f mergesort *.qdrep *.sqlite
	rm -rf vtune-report* mergesort-report* $(NCU_OUTPUT).ncu-rep
	rm -rf r000ps
