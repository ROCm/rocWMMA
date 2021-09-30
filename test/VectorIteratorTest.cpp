/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2021 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/
#include <hip/hip_runtime.h>

#include "Constants.h"
#include "Types.h"
#include "Utils.h"
#include <random>
#include <type_traits>
#include <unistd.h>
#include <utility>

#include "WMMA.h"
#include <gtest/gtest.h>

#include "Common.hpp"

template <typename DataT, uint32_t VecSize, uint32_t subVecSize>
__global__ void defaultConstructorTest()
{
    VecT<DataT, VecSize> dataVector;
    static_assert(dataVector.size() == VecSize, " Allocation Error");
}

template <typename DataT, uint32_t VecSize, uint32_t subVecSize>
__global__ void otherConstructorTest()
{
    VecT<DataT, VecSize> dataVector;
    static_assert(dataVector.size() == VecSize, " Allocation Error");
    VecT<DataT, VecSize>* otherDataVector(&dataVector);
    assert(otherDataVector != NULL);
    assert(otherDataVector->size() == dataVector.size());
}

template <typename DataT, uint32_t VecSize, uint32_t subVecSize>
__global__ void index()
{
    VecT<DataT, VecSize> dataVector;
    static_assert(dataVector.size() == VecSize, " Allocation Error");
    DataT val;
    for(uint32_t i = 0; i < VecSize; i++)
        val = dataVector[i];
}

template <typename DataT, uint32_t VecSize, uint32_t subVecSize>
__global__ void dereference()
{
    VecT<DataT, VecSize> dataVector;
    static_assert(dataVector.size() == VecSize, " Allocation Error");
    VecT<DataT, VecSize>* otherDataVector(&dataVector);
    assert(otherDataVector != NULL);
    VecT<DataT, VecSize> val = *dataVector;
    static_assert(val.size() == VecSize, " Allocation Error");
}

template <typename DataT, uint32_t VecSize, uint32_t subVecSize>
__global__ void getSize()
{
    VecT<DataT, VecSize> dataVector;
    static_assert(dataVector.size() == VecSize, " Allocation Error");
}

template <typename DataT, uint32_t VecSize, uint32_t subVecSize>
__global__ void vectIterator()
{
    VecT<DataT, VecSize> dataVector;
    static_assert(dataVector.size() == VecSize, " Allocation Error");
    auto it = typename VecT<DataT, VecSize>::template Iterator<subVecSize>(dataVector);
    assert(it.valid());
}

template <typename DataT, uint32_t VecSize, uint32_t subVecSize>
__global__ void vectIteratorAccess()
{
    VecT<DataT, VecSize> dataVector;
    static_assert(dataVector.size() == VecSize, " Allocation Error");
    auto it = typename VecT<DataT, VecSize>::template Iterator<subVecSize>(dataVector);
    assert(sizeof(dataVector) == sizeof(*it));
}

template <typename DataT, uint32_t VecSize, uint32_t subVecSize>
__global__ void vectIteratorInc()
{
    VecT<DataT, VecSize> dataVector;
    static_assert(dataVector.size() == VecSize, " Allocation Error");
    auto it = typename VecT<DataT, VecSize>::template Iterator<subVecSize>(dataVector);
    assert(sizeof(dataVector) == sizeof(*it));
    for(int i = 0; i < VecSize; i += subVecSize)
    {
        assert(it.valid());
        assert(it.mParent[0] == dataVector[i]);
        it++;
    }
}

template <typename DataT, uint32_t VecSize, uint32_t subVecSize>
__global__ void vectIteratorDec()
{
    VecT<DataT, VecSize> dataVector;
    static_assert(dataVector.size() == VecSize, " Allocation Error");
    auto it = dataVector.template end<subVecSize>();
    assert(sizeof(dataVector) == sizeof(*it));
    for(int i = VecSize - 1; i >= 0; i -= subVecSize)
    {
        assert(it.valid());
        assert(it.mParent[0] == dataVector[i]);
        it--;
    }
}

template <typename DataT, uint32_t VecSize, uint32_t subVecSize>
__global__ void vectIteratorNext()
{
    VecT<DataT, VecSize> dataVector;
    static_assert(dataVector.size() == VecSize, " Allocation Error");
    auto it = typename VecT<DataT, VecSize>::template Iterator<subVecSize>(dataVector);
    assert(sizeof(dataVector) == sizeof(*it));
    for(int i = 0; i < VecSize; i += subVecSize)
    {
        assert(it.valid());
        assert(it.mParent[0] == dataVector[i]);
        it.next();
    }
}

template <typename DataT, uint32_t VecSize, uint32_t subVecSize>
__global__ void vectIteratorPrev()
{
    VecT<DataT, VecSize> dataVector;
    static_assert(dataVector.size() == VecSize, " Allocation Error");
    auto it = dataVector.template end<subVecSize>();
    assert(sizeof(dataVector) == sizeof(*it));
    for(int i = VecSize - 1; i >= 0; i -= subVecSize)
    {
        assert(it.valid());
        assert(it.mParent[0] == dataVector[i]);
        it.prev();
    }
}

template <typename DataT, uint32_t VecSize, uint32_t SubVecSize>
struct Kernel
{
public:
    Kernel()
    {
        gridDim  = dim3(1);
        blockDim = dim3(1);
    }

    void constructorWrapper()
    {
        hipLaunchKernelGGL((defaultConstructorTest<DataT, VecSize, SubVecSize>),
                           gridDim,
                           blockDim,
                           0, // sharedMemBytes
                           0); // stream

        hipLaunchKernelGGL((otherConstructorTest<DataT, VecSize, SubVecSize>),
                           gridDim,
                           blockDim,
                           0, // sharedMemBytes
                           0); // stream
    }

    void accessWrapper()
    {
        hipLaunchKernelGGL((index<DataT, VecSize, SubVecSize>),
                           gridDim,
                           blockDim,
                           0, // sharedMemBytes
                           0); // stream

        hipLaunchKernelGGL((dereference<DataT, VecSize, SubVecSize>),
                           gridDim,
                           blockDim,
                           0, // sharedMemBytes
                           0); // stream
    }

    void sizeWrapper()
    {
        hipLaunchKernelGGL((getSize<DataT, VecSize, SubVecSize>),
                           gridDim,
                           blockDim,
                           0, // sharedMemBytes
                           0); // stream
    }

    void iteratorWrapper()
    {
        hipLaunchKernelGGL((vectIterator<DataT, VecSize, SubVecSize>),
                           gridDim,
                           blockDim,
                           0, // sharedMemBytes
                           0); // stream

        hipLaunchKernelGGL((vectIteratorAccess<DataT, VecSize, SubVecSize>),
                           gridDim,
                           blockDim,
                           0, // sharedMemBytes
                           0); // stream

        hipLaunchKernelGGL((vectIteratorInc<DataT, VecSize, SubVecSize>),
                           gridDim,
                           blockDim,
                           0, // sharedMemBytes
                           0); // stream

        hipLaunchKernelGGL((vectIteratorDec<DataT, VecSize, SubVecSize>),
                           gridDim,
                           blockDim,
                           0, // sharedMemBytes
                           0); // stream

        hipLaunchKernelGGL((vectIteratorPrev<DataT, VecSize, SubVecSize>),
                           gridDim,
                           blockDim,
                           0, // sharedMemBytes
                           0); // stream

        hipLaunchKernelGGL((vectIteratorNext<DataT, VecSize, SubVecSize>),
                           gridDim,
                           blockDim,
                           0, // sharedMemBytes
                           0); // stream
    }

    ~Kernel() {}

private:
    dim3 gridDim, blockDim;
};

template <typename T>
struct VectorIteratorWrapper;

template <typename DataT, typename VecSize, typename SubVecSize>
struct VectorIteratorWrapper<std::tuple<DataT, VecSize, SubVecSize>> : public testing::Test
{
    Kernel<DataT, VecSize::value, SubVecSize::value>* obj;

    void SetUp() override
    {
        obj = new Kernel<DataT, VecSize::value, SubVecSize::value>();
    }

    void ConstructorSetup()
    {
        obj->constructorWrapper();
    }

    void AccessSetup()
    {
        obj->accessWrapper();
    }

    void SizeSetup()
    {
        obj->sizeWrapper();
    }

    void IteratorSetup()
    {
        obj->iteratorWrapper();
    }

    void TearDown() override
    {
        delete obj;
    }
};

using Implementations = testing::Types<
    // DataT, VecSize, SubVecSize
    std::tuple<float32_t, I<1024>, I<128>>,
    std::tuple<hfloat16_t, I<1024>, I<128>>,
    std::tuple<float16_t, I<1024>, I<128>>,
    std::tuple<int8_t, I<1024>, I<128>>,
    std::tuple<int32_t, I<1024>, I<128>>,
    std::tuple<uint8_t, I<1024>, I<128>>,
    std::tuple<uint32_t, I<1024>, I<128>>>;

TYPED_TEST_SUITE(VectorIteratorWrapper, Implementations);

TYPED_TEST(VectorIteratorWrapper, constructor)
{
    this->ConstructorSetup();
}

TYPED_TEST(VectorIteratorWrapper, access)
{
    this->AccessSetup();
}

TYPED_TEST(VectorIteratorWrapper, size)
{
    this->SizeSetup();
}

TYPED_TEST(VectorIteratorWrapper, iterator)
{
    this->IteratorSetup();
}
