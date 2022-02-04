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

#ifndef WMMA_DEVICE_VECTOR_ITERATOR_H
#define WMMA_DEVICE_VECTOR_ITERATOR_H

#include "Types.h"

namespace rocwmma
{

    template <uint32_t BlockM, typename DataT>
    __device__ static inline bool defaultConstructorTest()
    {
        // defaultConstructorTest
        VecT<DataT, BlockM> vectData;
        return (vectData.size() == BlockM);
    }

    template <uint32_t BlockM, typename DataT>
    __device__ static inline bool copyConstructorTest()
    {
        // copyConstructorTest0
        VecT<DataT, BlockM> vectData;
        static_assert(vectData.size() == BlockM, "Allocation Error");
        for(uint32_t i = 0; i < BlockM; i++)
            vectData[i] = DataT(i);

        VecT<DataT, BlockM>* copyVectData(&vectData);
        bool ret = (copyVectData != NULL) & (copyVectData->size() == vectData.size());
        for(uint32_t i = 0; i < copyVectData->size(); i++)
            ret &= ((*copyVectData)[i] == vectData[i]);

        return ret;
    }

    template <uint32_t BlockM, typename DataT>
    __device__ static inline bool dereferenceTest()
    {
        // dereferenceTest
        VecT<DataT, BlockM> vectData;
        static_assert(vectData.size() == BlockM, "Allocation Error");
        for(uint32_t i = 0; i < BlockM; i++)
            vectData[i] = DataT(i);

        VecT<DataT, BlockM>* copyVectData(&vectData);
        assert((copyVectData != NULL) && (copyVectData->size() == vectData.size()));

        VecT<DataT, BlockM> derefVectData = *copyVectData;
        bool                ret           = (derefVectData.size() == copyVectData->size());
        for(uint32_t i = 0; i < derefVectData.size(); i++)
            ret &= ((*copyVectData)[i] == derefVectData[i]);

        return ret;
    }

    template <uint32_t BlockM, typename DataT>
    __device__ static inline bool iteratorTest()
    {
        // iteratorTest
        VecT<DataT, BlockM> iterVectData;
        static_assert(iterVectData.size() == BlockM, " Allocation Error");
        for(uint32_t i = 0; i < BlockM; i++)
            iterVectData[i] = DataT(i);

        const uint32_t iterSize = BlockM / 2;
        auto           it = typename VecT<DataT, BlockM>::template iterator<iterSize>(iterVectData);
        bool           ret = (iterVectData.size() == (it.range() * iterSize));

        for(uint32_t i = 0; i < it.range(); i++, it++)
            ret &= (iterVectData[i * iterSize] == iterVectData[(*it)[0]]);

        return ret;
    }

    template <uint32_t BlockM, typename DataT>
    __device__ static inline bool iteratorValidityTest()
    {
        // iteratorValidityTest
        VecT<DataT, BlockM> iterVectData;
        static_assert(iterVectData.size() == BlockM, " Allocation Error");
        for(uint32_t i = 0; i < BlockM; i++)
            iterVectData[i] = DataT(i);

        const uint32_t iterSize = BlockM / 2;
        auto           it = typename VecT<DataT, BlockM>::template iterator<iterSize>(iterVectData);
        bool           ret = it.valid();

        ret &= (iterVectData.size() == (it.range() * iterSize));

        return ret;
    }

    template <uint32_t BlockM, typename DataT>
    __device__ static inline bool iteratorIndexTest()
    {
        // iteratorIndexTest
        VecT<DataT, BlockM> iterVectData;
        static_assert(iterVectData.size() == BlockM, " Allocation Error");
        for(uint32_t i = 0; i < BlockM; i++)
            iterVectData[i] = DataT(i);

        const uint32_t iterSize = BlockM / 2;
        auto           it = typename VecT<DataT, BlockM>::template iterator<iterSize>(iterVectData);
        bool           ret = it.valid();

        ret &= (iterVectData.size() == (it.range() * iterSize));
        ret &= (it.index() == 0);

        auto nextit = it.next();

        ret &= (nextit.valid());
        ret &= (nextit.index() == 1);

        return ret;
    }

    template <uint32_t BlockM, typename DataT>
    __device__ static inline bool iteratorRangeTest()
    {
        // iteratorRangeTest
        VecT<DataT, BlockM> iterVectData;
        static_assert(iterVectData.size() == BlockM, " Allocation Error");
        for(uint32_t i = 0; i < BlockM; i++)
            iterVectData[i] = DataT(i);

        const uint32_t iterSize = BlockM / 2;
        auto           it = typename VecT<DataT, BlockM>::template iterator<iterSize>(iterVectData);
        bool           ret = it.valid();

        ret &= ((iterVectData.size() / iterSize) == it.range());

        return ret;
    }

    template <uint32_t BlockM, typename DataT>
    __device__ static inline bool iteratorBeginTest()
    {
        // iteratorBeginTest
        VecT<DataT, BlockM> iterVectData;
        static_assert(iterVectData.size() == BlockM, " Allocation Error");
        for(uint32_t i = 0; i < BlockM; i++)
            iterVectData[i] = DataT(i);

        const uint32_t iterSize = BlockM / 2;
        auto           it       = iterVectData.template begin<iterSize>();
        assert(iterVectData.size() == (it.range() * iterSize));

        bool ret = true;

        ret &= (it.valid());
        ret &= (iterVectData[0] == iterVectData[(*it)[0]]);

        return ret;
    }

    template <uint32_t BlockM, typename DataT>
    __device__ static inline bool iteratorEndTest()
    {
        // iteratorEndTest
        VecT<DataT, BlockM> iterVectData;
        static_assert(iterVectData.size() == BlockM, " Allocation Error");
        for(uint32_t i = 0; i < BlockM; i++)
            iterVectData[i] = DataT(i);

        const uint32_t iterSize = BlockM / 2;
        auto           it       = iterVectData.template end<iterSize>();
        assert(iterVectData.size() == (it.range() * iterSize));

        bool ret = true;

        ret &= (iterVectData[0] == iterVectData[(*it)[0]]);

        return ret;
    }

    template <uint32_t BlockM, typename DataT>
    __device__ static inline bool iteratorObjTest()
    {
        // iteratorEndTest
        VecT<DataT, BlockM> iterVectData;
        static_assert(iterVectData.size() == BlockM, " Allocation Error");
        for(uint32_t i = 0; i < BlockM; i++)
            iterVectData[i] = DataT(i);

        const uint32_t iterSize = BlockM / 2;
        auto           it       = iterVectData.template it<iterSize>();
        assert(iterVectData.size() == (it.range() * iterSize));

        bool ret = true;

        ret &= (iterVectData[0] == iterVectData[(*it)[0]]);

        return ret;
    }

    template <uint32_t BlockM, typename DataT>
    __device__ static inline bool iteratorIncTest()
    {
        // iteratorIncTest
        VecT<DataT, BlockM> iterVectData;
        static_assert(iterVectData.size() == BlockM, " Allocation Error");
        for(uint32_t i = 0; i < BlockM; i++)
            iterVectData[i] = DataT(i);

        const uint32_t iterSize = BlockM / 2;
        auto           it = typename VecT<DataT, BlockM>::template iterator<iterSize>(iterVectData);
        assert(iterVectData.size() == (it.range() * iterSize));

        bool ret = true;
        for(uint32_t i = 0; i < it.range(); i++)
        {
            ret &= (it.valid());
            ret &= (iterVectData[i * iterSize] == iterVectData[(*it)[0]]);
            ++it;
        }

        return ret;
    }

    template <uint32_t BlockM, typename DataT>
    __device__ static inline bool iteratorDecTest()
    {
        // iteratorDecTest
        VecT<DataT, BlockM> iterVectData;
        static_assert(iterVectData.size() == BlockM, " Allocation Error");
        for(uint32_t i = 0; i < BlockM; i++)
            iterVectData[i] = DataT(i);

        const uint32_t iterSize = BlockM / 2;
        auto           it       = iterVectData.template end<iterSize>();
        assert(iterVectData.size() == (it.range() * iterSize));

        bool ret = true;
        for(uint32_t i = 0; i < it.range(); i++)
        {
            ret &= (iterVectData[i * iterSize] == iterVectData[(*it)[0]]);
            --it;
            ret &= (it.valid());
        }

        return ret;
    }

    template <uint32_t BlockM, typename DataT>
    __device__ static inline bool iteratorNextTest()
    {
        // iteratorNextTest
        VecT<DataT, BlockM> iterVectData;
        static_assert(iterVectData.size() == BlockM, " Allocation Error");
        for(uint32_t i = 0; i < BlockM; i++)
            iterVectData[i] = DataT(i);

        const uint32_t iterSize = BlockM / 2;
        auto           it = typename VecT<DataT, BlockM>::template iterator<iterSize>(iterVectData);
        assert(iterVectData.size() == (it.range() * iterSize));

        bool ret = true;

        ret &= (it.valid());
        ret &= (iterVectData[0] == iterVectData[(*it)[0]]);
        auto nextit = it.next();

        ret &= (nextit.valid());
        ret &= (iterVectData[iterSize] == iterVectData[(*nextit)[0]]);

        return ret;
    }

    template <uint32_t BlockM, typename DataT>
    __device__ static inline bool iteratorPrevTest()
    {
        // iteratorPrevTest
        VecT<DataT, BlockM> iterVectData;
        static_assert(iterVectData.size() == BlockM, " Allocation Error");
        for(uint32_t i = 0; i < BlockM; i++)
            iterVectData[i] = DataT(i);

        const uint32_t iterSize = BlockM / 2;
        auto           it       = iterVectData.template end<iterSize>();
        assert(iterVectData.size() == (it.range() * iterSize));

        bool ret = true;

        ret &= (iterVectData[0] == iterVectData[(*it)[0]]);
        auto previt = it.prev();

        ret &= (previt.valid());
        ret &= (iterVectData[iterSize] == iterVectData[(*previt)[0]]);

        return ret;
    }

    template <uint32_t BlockM, uint32_t BlockN, typename DataT, typename Layout>
    __global__ void VectorIterator(uint32_t     m,
                                   uint32_t     n,
                                   DataT const* in,
                                   DataT*       out,
                                   uint32_t     ld,
                                   DataT        param1,
                                   DataT        param2)
    {

        if(threadIdx.x == 0 && threadIdx.y == 0)
        {
            bool err = defaultConstructorTest<BlockM, DataT>();
            assert(err == true);

            err &= copyConstructorTest<BlockM, DataT>();
            assert(err == true);

            err &= dereferenceTest<BlockM, DataT>();
            assert(err == true);

            err &= iteratorTest<BlockM, DataT>();
            assert(err == true);

            err &= iteratorValidityTest<BlockM, DataT>();
            assert(err == true);

            err &= iteratorIndexTest<BlockM, DataT>();
            assert(err == true);

            err &= iteratorRangeTest<BlockM, DataT>();
            assert(err == true);

            err &= iteratorBeginTest<BlockM, DataT>();
            assert(err == true);

            err &= iteratorEndTest<BlockM, DataT>();
            assert(err == true);

            err &= iteratorObjTest<BlockM, DataT>();
            assert(err == true);

            err &= iteratorIncTest<BlockM, DataT>();
            assert(err == true);

            err &= iteratorDecTest<BlockM, DataT>();
            assert(err == true);

            err &= iteratorNextTest<BlockM, DataT>();
            assert(err == true);

            err &= iteratorPrevTest<BlockM, DataT>();
            assert(err == true);
        }
    }

} // namespace rocwmma

#endif // WMMA_DEVICE_VECTOR_ITERATOR_H
