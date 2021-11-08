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

#ifndef WMMA_DEVICE_VECTOR_ITERATOR_H
#define WMMA_DEVICE_VECTOR_ITERATOR_H

#include "Types.h"

template <uint32_t BlockM, uint32_t BlockN, typename DataT, typename Layout>
__global__ void VectorIterator(
    uint32_t m, uint32_t n, DataT const* in, DataT* out, uint32_t ld, DataT param1, DataT param2)
{
    // defaultConstructorTest
    VecT<DataT, BlockM> vectD;
    static_assert(vectD.size() == BlockM, " Allocation Error");

    // otherConstructorTest
    VecT<DataT, BlockM>* otherVectD(&vectD);
    assert(otherVectD != NULL);
    assert(otherVectD->size() == vectD.size());

    // index
    DataT ind;
    for(uint32_t i = 0; i < BlockM; i++)
        ind = vectD[i];

    // dereference
    VecT<DataT, BlockM> deref = *otherVectD;
    static_assert(deref.size() == BlockM, " Allocation Error");

    // vectIteratorAccess
    VecT<DataT, BlockM> accessVectD;
    static_assert(accessVectD.size() == BlockM, " Allocation Error");
    auto iter = typename VecT<DataT, BlockM>::template Iterator<BlockM / 2>(accessVectD);
    assert(accessVectD.size() == sizeof(iter));

    // vectIteratorInc
    VecT<DataT, BlockM> incVectD;
    for(uint32_t i = 0; i < BlockM; i++)
        incVectD[i] = DataT(i);
    auto iti = typename VecT<DataT, BlockM>::template Iterator<BlockM / 2>(incVectD);
    assert(incVectD.size() == sizeof(iti));
    for(int i = 0; i < BlockM; i += BlockM / 2)
    {
        assert(iti.valid());
        assert(iti.mParent[i] == incVectD[(*iti)[0]]);
        iti++;
    }

    // vectIteratorDec
    VecT<DataT, BlockM> decVectD;
    for(uint32_t i = 0; i < BlockM; i++)
        decVectD[i] = DataT(i);
    auto itd = decVectD.template end<BlockM / 2>();
    assert(decVectD.size() == sizeof(itd));
    for(int i = 0; i < BlockM; i += (BlockM / 2))
    {
        --itd;
        assert(itd.valid());
        assert(itd.mParent[(BlockM / 2) - i] == decVectD[(*itd)[0]]);
    }

    // vectIteratorNext
    VecT<DataT, BlockM> nextVectD;
    for(uint32_t i = 0; i < BlockM; i++)
        nextVectD[i] = DataT(i);
    auto itn = typename VecT<DataT, BlockM>::template Iterator<BlockM / 2>(nextVectD);
    assert(itn.valid());
    assert(nextVectD.size() == sizeof(itn));
    auto nextitn = itn.next();
    assert(nextitn.valid());
    assert(nextitn.mParent[(BlockM / 2)] == nextVectD[(*nextitn)[0]]);

    // vectIteratorPrev
    VecT<DataT, BlockM> prevVectD;
    for(uint32_t i = 0; i < BlockM; i++)
        prevVectD[i] = DataT(i);
    auto itp = prevVectD.template end<BlockM / 2>();
    assert(prevVectD.size() == sizeof(itp));
    auto previtp = itp.prev();
    assert(previtp.valid());
    assert(previtp.mParent[BlockM - (BlockM / 2)] == prevVectD[(*previtp)[0]]);
}

#endif // WMMA_DEVICE_VECTOR_ITERATOR_H
