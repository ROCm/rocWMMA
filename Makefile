CXX=/opt/rocm/hip/bin/hipcc

CXXFLAGS=-I/opt/rocm/hip/include/ -march=native

ALL = BufferLoadTest

all: $(ALL)

$(TARGET): $(TARGET:.cpp)
	$(CXX) $(CXXFLAGS) -o $@ $^

.PHONY: clean
clean:
	rm -f *.o $(ALL)
