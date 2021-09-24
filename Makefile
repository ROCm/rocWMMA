###############################################################################
 #
 # MIT License
 #
 # Copyright 2021 Advanced Micro Devices, Inc.
 #
 # Permission is hereby granted, free of charge, to any person obtaining a copy
 # of this software and associated documentation files (the "Software"), to deal
 # in the Software without restriction, including without limitation the rights
 # to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 # copies of the Software, and to permit persons to whom the Software is
 # furnished to do so, subject to the following conditions:
 #
 # The above copyright notice and this permission notice shall be included in
 # all copies or substantial portions of the Software.
 #
 # THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 # IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 # FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 # AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 # LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 # OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 # SOFTWARE.
 #
 ###############################################################################
CXX=/opt/rocm/hip/bin/hipcc

CXXFLAGS=-I/opt/rocm/hip/include/ -I./library/include -march=native -O3 -std=c++14
ALL = test/FillFragmentTest test/LoadStoreMatrixSyncTest test/MmaSyncTest-bench test/MmaSyncTest-cpu

all: $(ALL)

MmaSyncTest-bench: test/MmaSyncTest.cpp
	$(CXX) $(CXXFLAGS) -o MmaSyncTest-bench $^

MmaSyncTest-cpu: test/MmaSyncTest.cpp
	$(CXX) $(CXXFLAGS) -DWMMA_VALIDATE_TESTS -o MmaSyncTest-cpu $^

MmaSyncTest-rocBLAS: test/MmaSyncTest.cpp
	$(CXX) $(CXXFLAGS) -DWMMA_VALIDATE_WITH_ROCBLAS -DWMMA_VALIDATE_TESTS -I$(ROCBLAS_DIR)/include -I$(ROCBLAS_DIR)/include/internal -L$(ROCBLAS_DIR)/lib -lrocblas -ldl -o MmaSyncTest-rocBLAS $^

LoadStoreMatrixSyncTest: test/LoadStoreMatrixSyncTest.cpp
	$(CXX) $(CXXFLAGS) -DWMMA_VALIDATE_TESTS -o LoadStoreMatrixSyncTest $^

$(TARGET): $(TARGET:.cpp)
	$(CXX) $(CXXFLAGS) -DWMMA_VALIDATE_TESTS -o $@ $^

.PHONY: clean
clean:
	rm -f *.o $(ALL) MmaSyncTest-rocBLAS
