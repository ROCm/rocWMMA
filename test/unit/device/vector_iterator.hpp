/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2021-2022 Advanced Micro Devices, Inc.
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

#ifndef ROCWMMA_DEVICE_VECTOR_ITERATOR_HPP
#define ROCWMMA_DEVICE_VECTOR_ITERATOR_HPP

#include <rocwmma/internal/types.hpp>

#include <../test/gemm/gemm_coop_schedule.hpp>
#include <rocwmma/internal/mapping_util.hpp>

static constexpr uint32_t ERROR_VALUE = 7;
static constexpr uint32_t SUCCESS     = 0;

namespace rocwmma
{

    template <uint32_t VectSize, typename DataT>
    __device__ static inline bool somethingTest()
    {
        // defaultConstructorTest
        constexpr non_native_vector_base<uint32_t, 2> coord{2u, 2u};
        constexpr non_native_vector_base<uint32_t, 2> coord2{2u, 3u};
        constexpr uint32_t item = CooperativeGemm::Schedule::SameRowFwd<64, 1>::waveCount();
        return (false);
    }

    template <uint32_t VectSize, typename DataT>
    __device__ static inline bool defaultConstructorTest()
    {
        // defaultConstructorTest
        VecT<DataT, VectSize> vectData;
        return (vectData.size() == VectSize);
    }

    template <uint32_t VectSize, typename DataT>
    __device__ static inline bool copyConstructorTest()
    {
        // copyConstructorTest0
        VecT<DataT, VectSize> vectData;
        bool                  ret = (vectData.size() == VectSize);

        for(uint32_t i = 0; i < VectSize; i++)
            vectData[i] = static_cast<DataT>(i);

        VecT<DataT, VectSize> copyVectData(vectData);
        ret &= (copyVectData.size() == vectData.size());
        for(uint32_t i = 0; i < copyVectData.size(); i++)
            ret &= (copyVectData[i] == vectData[i]);

        VecT<DataT, VectSize> copyStorageData(*vectData);
        ret &= (copyStorageData.size() == vectData.size());
        for(uint32_t i = 0; i < copyStorageData.size(); i++)
            ret &= (copyStorageData[i] == vectData[i]);

        VecT<DataT, VectSize> moveStorageData(std::move(*vectData));
        ret &= (moveStorageData.size() == vectData.size());
        for(uint32_t i = 0; i < moveStorageData.size(); i++)
            ret &= (moveStorageData[i] == vectData[i]);

        return ret;
    }

    template <uint32_t VectSize, typename DataT>
    __device__ static inline bool dereferenceTest()
    {
        // dereferenceTest
        VecT<DataT, VectSize> vectData;
        bool                  ret = (vectData.size() == VectSize);
        for(uint32_t i = 0; i < VectSize; i++)
            vectData[i] = static_cast<DataT>(i);

        VecT<DataT, VectSize> copyVectData(vectData);

        typename VecT<DataT, VectSize>::StorageT storageT = *copyVectData;
        ret &= (sizeof(storageT) == (sizeof(DataT) * vectData.size()));

        for(uint32_t i = 0; i < copyVectData.size(); i++)
            ret &= (storageT[i] == copyVectData[i]);

        return ret;
    }

    template <uint32_t VectSize, typename DataT>
    __device__ static inline bool iteratorTest()
    {
        // iteratorTest
        VecT<DataT, VectSize> iterVectData;
        bool                  ret = (iterVectData.size() == VectSize);
        for(uint32_t i = 0; i < VectSize; i++)
            iterVectData[i] = static_cast<DataT>(i);

        const uint32_t iterSize = VectSize / 2;
        auto it = typename VecT<DataT, VectSize>::template iterator<iterSize>(iterVectData);
        ret &= (iterVectData.size() == (it.range() * iterSize));

        for(uint32_t i = 0; i < it.range(); i++, it++)
        {
            for(uint32_t j = 0; j < iterSize; j++)
            {
                ret &= (iterVectData[i * iterSize + j] == iterVectData[(*it)[j]]);
            }
        }

        return ret;
    }

    template <uint32_t VectSize, typename DataT>
    __device__ static inline bool iteratorValidityTest()
    {
        // iteratorValidityTest
        VecT<DataT, VectSize> iterVectData;
        bool                  ret = (iterVectData.size() == VectSize);
        for(uint32_t i = 0; i < VectSize; i++)
            iterVectData[i] = static_cast<DataT>(i);

        const uint32_t iterSize = VectSize / 2;
        auto it = typename VecT<DataT, VectSize>::template iterator<iterSize>(iterVectData);
        ret &= it.valid();

        ret &= (iterVectData.size() == (it.range() * iterSize));

        return ret;
    }

    template <uint32_t VectSize, typename DataT>
    __device__ static inline bool iteratorIndexTest()
    {
        // iteratorIndexTest
        VecT<DataT, VectSize> iterVectData;
        bool                  ret = (iterVectData.size() == VectSize);
        for(uint32_t i = 0; i < VectSize; i++)
            iterVectData[i] = static_cast<DataT>(i);

        const uint32_t iterSize = VectSize / 2;
        auto it = typename VecT<DataT, VectSize>::template iterator<iterSize>(iterVectData);
        ret &= it.valid();

        ret &= (iterVectData.size() == (it.range() * iterSize));
        ret &= (it.index() == 0);

        auto nextit = it.next();

        ret &= (nextit.valid());
        ret &= (nextit.index() == 1);

        return ret;
    }

    template <uint32_t VectSize, typename DataT>
    __device__ static inline bool iteratorRangeTest()
    {
        // iteratorRangeTest
        VecT<DataT, VectSize> iterVectData;
        bool                  ret = (iterVectData.size() == VectSize);
        for(uint32_t i = 0; i < VectSize; i++)
            iterVectData[i] = static_cast<DataT>(i);

        const uint32_t iterSize = VectSize / 2;
        auto it = typename VecT<DataT, VectSize>::template iterator<iterSize>(iterVectData);
        ret &= it.valid();

        ret &= ((iterVectData.size() / iterSize) == it.range());

        return ret;
    }

    template <uint32_t VectSize, typename DataT>
    __device__ static inline bool iteratorBeginEndTest()
    {
        // iteratorBeginTest
        VecT<DataT, VectSize> iterVectData;
        bool                  ret = (iterVectData.size() == VectSize);
        for(uint32_t i = 0; i < VectSize; i++)
            iterVectData[i] = static_cast<DataT>(i);

        const uint32_t iterSize = VectSize / 2;
        auto           itBegin  = iterVectData.template begin<iterSize>();
        ret &= (iterVectData.size() == (itBegin.range() * iterSize));

        ret &= (itBegin.valid());
        ret &= (iterVectData[0] == iterVectData[(*itBegin)[0]]);

        auto itEnd = iterVectData.template end<iterSize>();
        ret &= (iterVectData.size() == (itEnd.range() * iterSize));

        ret &= (iterVectData[0] == iterVectData[(*itEnd)[0]]);

        return ret;
    }

    template <uint32_t VectSize, typename DataT>
    __device__ static inline bool iteratorObjTest()
    {
        // iteratorEndTest
        VecT<DataT, VectSize> iterVectData;
        bool                  ret = (iterVectData.size() == VectSize);
        for(uint32_t i = 0; i < VectSize; i++)
            iterVectData[i] = static_cast<DataT>(i);

        const uint32_t iterSize = VectSize / 2;
        auto           it       = iterVectData.template it<iterSize>();
        ret &= (iterVectData.size() == (it.range() * iterSize));

        ret &= (iterVectData[0] == iterVectData[(*it)[0]]);

        return ret;
    }

    template <uint32_t VectSize, typename DataT>
    __device__ static inline bool iteratorIncDecTest()
    {
        // iteratorIncTest
        VecT<DataT, VectSize> iterVectData;
        bool                  ret = (iterVectData.size() == VectSize);
        for(uint32_t i = 0; i < VectSize; i++)
            iterVectData[i] = static_cast<DataT>(i);

        const uint32_t iterSize = VectSize / 2;

        auto itInc = typename VecT<DataT, VectSize>::template iterator<iterSize>(iterVectData);
        ret &= (iterVectData.size() == (itInc.range() * iterSize));
        for(uint32_t i = 0; i < itInc.range(); i++)
        {
            ret &= (itInc.valid());
            ret &= (iterVectData[i * iterSize] == iterVectData[(*itInc)[0]]);
            ++itInc;
        }

        auto itDec = iterVectData.template end<iterSize>();
        ret &= (iterVectData.size() == (itDec.range() * iterSize));
        for(uint32_t i = 0; i < itDec.range(); i++)
        {
            ret &= (iterVectData[i * iterSize] == iterVectData[(*itDec)[0]]);
            --itDec;
            ret &= (itDec.valid());
        }

        return ret;
    }

    template <uint32_t VectSize, typename DataT>
    __device__ static inline bool iteratorNextPrevTest()
    {
        // iteratorNextTest
        VecT<DataT, VectSize> iterVectData;
        bool                  ret = (iterVectData.size() == VectSize);
        for(uint32_t i = 0; i < VectSize; i++)
            iterVectData[i] = static_cast<DataT>(i);

        const uint32_t iterSize = VectSize / 2;
        auto it = typename VecT<DataT, VectSize>::template iterator<iterSize>(iterVectData);
        ret &= (iterVectData.size() == (it.range() * iterSize));

        ret &= (it.valid());
        ret &= (iterVectData[0] == iterVectData[(*it)[0]]);

        auto nextit = it.next();
        ret &= (nextit.valid());
        ret &= (iterVectData[iterSize] == iterVectData[(*nextit)[0]]);

        auto previt = nextit.prev();
        ret &= (previt.valid());
        ret &= (iterVectData[0] == iterVectData[(*previt)[0]]);
        return ret;
    }

    template <uint32_t VectSize, typename DataT>
    __global__ void VectorIterator(uint32_t     m,
                                   uint32_t     n,
                                   DataT const* in,
                                   DataT*       out,
                                   uint32_t     ld,
                                   DataT        param1,
                                   DataT        param2)
    {
        // Just need one thread
        if(threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0 && blockIdx.x == 0
           && blockIdx.y == 0 && blockIdx.z == 0)
        {
            out[0] = static_cast<DataT>(SUCCESS);

            bool err = defaultConstructorTest<VectSize, DataT>();
            if(err == false)
            {
                out[0] = static_cast<DataT>(ERROR_VALUE);
                return;
            }

            err &= copyConstructorTest<VectSize, DataT>();
            if(err == false)
            {
                out[0] = static_cast<DataT>(ERROR_VALUE);
                return;
            }

            err &= dereferenceTest<VectSize, DataT>();
            if(err == false)
            {
                out[0] = static_cast<DataT>(ERROR_VALUE);
                return;
            }

            err &= iteratorTest<VectSize, DataT>();
            if(err == false)
            {
                out[0] = static_cast<DataT>(ERROR_VALUE);
                return;
            }

            err &= iteratorValidityTest<VectSize, DataT>();
            if(err == false)
            {
                out[0] = static_cast<DataT>(ERROR_VALUE);
                return;
            }

            err &= iteratorIndexTest<VectSize, DataT>();
            if(err == false)
            {
                out[0] = static_cast<DataT>(ERROR_VALUE);
                return;
            }

            err &= iteratorRangeTest<VectSize, DataT>();
            if(err == false)
            {
                out[0] = static_cast<DataT>(ERROR_VALUE);
                return;
            }

            err &= iteratorBeginEndTest<VectSize, DataT>();
            if(err == false)
            {
                out[0] = static_cast<DataT>(ERROR_VALUE);
                return;
            }

            err &= iteratorObjTest<VectSize, DataT>();
            if(err == false)
            {
                out[0] = static_cast<DataT>(ERROR_VALUE);
                return;
            }

            err &= iteratorIncDecTest<VectSize, DataT>();
            if(err == false)
            {
                out[0] = static_cast<DataT>(ERROR_VALUE);
                return;
            }

            err &= iteratorNextPrevTest<VectSize, DataT>();
            if(err == false)
            {
                out[0] = static_cast<DataT>(ERROR_VALUE);
                return;
            }
        }
    }

} // namespace rocwmma

#endif // ROCWMMA_DEVICE_VECTOR_ITERATOR_HPP
