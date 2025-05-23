CUDACXX = nvcc
CUDACXXFLAGS = -O3 -lineinfo -Xptxas -v -Xcudafe "--display_error_number"
CXXFLAGS = -march=native

NSYS = nsys profile
NSYSFLAGS = --stats=true --force-overwrite=true

VTUNE = vtune
VTUNEFLAGS_CPU = -collect hotspots
VTUNEFLAGS_GPU = -collect gpu-hotspots
VTUNE_RESULT_DIR_CPU = vtune-report-cpu
VTUNE_RESULT_DIR_GPU = vtune-report-gpu

NCU = ncu
NCUFLAGS = --set full --replay-mode kernel --apply-rules yes
NCU_OUTPUT_PREFIX = pipeline_demo_ncu_report

# Detect GPU arch dynamically (fallback to sm_75)
GPU_ARCH := $(shell if command -v nvidia-smi >/dev/null 2>&1; then \
	nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1 | sed 's/\.//'; \
	else echo 75; fi)

GENCODE_FLAGS := -gencode arch=compute_$(GPU_ARCH),code=sm_$(GPU_ARCH)

all: pipeline_demo

pipeline_demo: pipeline_demo.cu
	$(CUDACXX) $(GENCODE_FLAGS) $(CUDACXXFLAGS) -Xcompiler="$(CXXFLAGS)" pipeline_demo.cu -o pipeline_demo

profile: pipeline_demo
	$(NSYS) $(NSYSFLAGS) -o pipeline_demo-report ./pipeline_demo

vtune-cpu: pipeline_demo
	rm -rf $(VTUNE_RESULT_DIR_CPU)
	$(VTUNE) $(VTUNEFLAGS_CPU) -result-dir $(VTUNE_RESULT_DIR_CPU) ./pipeline_demo
	$(VTUNE) -collect performance-snapshot -collect memory-access -collect hotspots -collect threading -- ./pipeline_demo

vtune-gpu: pipeline_demo
	rm -rf $(VTUNE_RESULT_DIR_GPU)
	@if command -v $(VTUNE) >/dev/null 2>&1; then \
		echo "Running VTune GPU profiling"; \
		$(VTUNE) $(VTUNEFLAGS_GPU) -result-dir $(VTUNE_RESULT_DIR_GPU) ./pipeline_demo; \
	else \
		echo "Warning: VTune is not installed or not in PATH. Skipping GPU profiling."; \
	fi


ncu: pipeline_demo
ifeq ($(shell [ $(GPU_ARCH) -ge 80 ] && echo true),true)
	@echo "Detected A100 or newer (sm_$(GPU_ARCH)), profiling each kernel separately"
else
	@echo "Detected T4 or older (sm_$(GPU_ARCH)), profiling each kernel separately"
endif
	$(NCU) $(NCUFLAGS) --kernel-name "copy_pipeline" -o $(NCU_OUTPUT_PREFIX)_pipeline ./pipeline_demo
	$(NCU) $(NCUFLAGS) --kernel-name "copy_manual" -o $(NCU_OUTPUT_PREFIX)_manual ./pipeline_demo


clean:
	rm -f pipeline_demo *.qdrep *.sqlite
	rm -rf vtune-report* pipeline_demo-report* $(NCU_OUTPUT_PREFIX)*.ncu-rep
	rm -rf r000ps