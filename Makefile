CXX=/opt/rocm/hip/bin/hipcc

CXXFLAGS=-I/opt/rocm/hip/include/ -I. -march=native -O3 -std=c++14
ALL = test/FillFragmentTest test/LoadStoreMatrixSyncTest test/MmaSyncTest-bench test/MmaSyncTest-cpu

all: $(ALL)

MmaSyncTest-bench: test/MmaSyncTest.cpp
	$(CXX) $(CXXFLAGS) -o MmaSyncTest-bench $^

MmaSyncTest-cpu: test/MmaSyncTest.cpp
	$(CXX) $(CXXFLAGS) -DWMMA_VALIDATE_TESTS -o MmaSyncTest-cpu $^

MmaSyncTest-rocBLAS: test/MmaSyncTest.cpp
	$(CXX) $(CXXFLAGS) -DWMMA_VALIDATE_WITH_ROCBLAS -DWMMA_VALIDATE_TESTS -I$(ROCBLAS_DIR)/include -I$(ROCBLAS_DIR)/include/internal -L$(ROCBLAS_DIR)/lib -lrocblas -o MmaSyncTest-rocBLAS $^

$(TARGET): $(TARGET:.cpp)
	$(CXX) $(CXXFLAGS) -DWMMA_VALIDATE_TESTS -o $@ $^

.PHONY: clean
clean:
	rm -f *.o $(ALL) MmaSyncTest-rocBLAS
