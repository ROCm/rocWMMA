#ifndef WMMA_CONSTANTS_H
#define WMMA_CONSTANTS_H

#include "Types.h"

static constexpr uint32_t AMDGCN_WAVE_SIZE   = 64;
static constexpr uint32_t BYTES_PER_REGISTER = 256;
static constexpr uint32_t MAX_ELEMENTS_PER_THREAD = 4;

static constexpr uint32_t LDS_MAX_BYTES = 65536;

template <typename DataT>
static constexpr uint32_t elementsPerRegister()
{
    return BYTES_PER_REGISTER / sizeof(DataT);
}

#endif // WMMA_CONSTANTS_H
