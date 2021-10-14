.DEFAULT_GOAL := hip

.PHONY: hip cuda clean check_hip_wmma

hip: check_hip_wmma
	hipcc -o dot_based_interact -std=c++14 -I$(HIP_WMMA_DIR) ./dot_based_interact.hip

cuda:
	nvcc -o dot_based_interact -gencode arch=compute_70,code=sm_70 dot_based_interact.cu

clean:
	rm -f dot_based_interact

check_hip_wmma:
ifndef HIP_WMMA_DIR
	$(error HIP_WMMA_DIR is not set.)
endif
