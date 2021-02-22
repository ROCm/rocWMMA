CXX=/opt/rocm/hip/bin/hipcc

CXXFLAGS=-I/opt/rocm/hip/include/ -march=native -O2 -DWMMA_VALIDATE_TESTS

ALL = FillFragmentTest LoadStoreMatrixSyncTest MmaSyncTest MmaSyncCoopTest

all: $(ALL)

$(TARGET): $(TARGET:.cpp)
	$(CXX) $(CXXFLAGS) -o $@ $^

.PHONY: clean
clean:
	rm -f *.o $(ALL)
