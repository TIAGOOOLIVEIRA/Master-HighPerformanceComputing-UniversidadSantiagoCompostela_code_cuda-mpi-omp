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
NCUFLAGS=--target-processes all --metrics sm__sass_average_branch_targets_threads_per_instruction.pct,sm__pipe_shared_mem_op_stall_cycles.avg.per_cycle_active,sm__inst_executed_pipe_shared_mem.sum --kernel-name "sort_kernel"
NCU_OUTPUT=mergesort_ncu

GPU_ARCH := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1 | sed 's/\.//')
GENCODE_FLAGS := -gencode arch=compute_$(GPU_ARCH),code=sm_$(GPU_ARCH)

all: mergesort

mergesort: mergesort.cu
	$(CUDACXX) $(GENCODE_FLAGS) $(CUDACXXFLAGS) -Xcompiler="$(CXXFLAGS)" mergesort.cu -o mergesort

profile: mergesort
	$(NSYS) $(NSYSFLAGS) -o mergesort-report ./mergesort

vtune-cpu: mergesort
	rm -rf $(VTUNE_RESULT_DIR_CPU)
	$(VTUNE) $(VTUNEFLAGS_CPU) -result-dir $(VTUNE_RESULT_DIR_CPU) ./mergesort

vtune-gpu: mergesort
	rm -rf $(VTUNE_RESULT_DIR_GPU)
	@if command -v $(VTUNE) >/dev/null 2>&1; then \
		echo "Running VTune GPU profiling..."; \
		$(VTUNE) $(VTUNEFLAGS_GPU) -result-dir $(VTUNE_RESULT_DIR_GPU) ./mergesort; \
	else \
		echo "Warning: VTune is not installed or not in PATH. Skipping GPU profiling."; \
	fi

ncu: mergesort
	$(NCU) $(NCUFLAGS) ./mergesort -o $(NCU_OUTPUT)

clean:
	rm -f mergesort *.qdrep *.sqlite
	rm -rf vtune-report* mergesort-report* $(NCU_OUTPUT).ncu-rep
