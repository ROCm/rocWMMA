#include "gemm_resource_impl.hpp"

namespace rocwmma
{
    // All supported instantiations
    template struct GemmResource<int8_t, int32_t>;
    template struct GemmResource<bfloat16_t, float32_t>;
    template struct GemmResource<float16_t, float32_t>;
    template struct GemmResource<hfloat16_t, float32_t>;
    template struct GemmResource<float32_t, float32_t>;
    template struct GemmResource<float64_t, float64_t>;

#if defined(ROCWMMA_EXTENDED_TESTS)
    template struct GemmResource<int8_t, int8_t>;
    template struct GemmResource<bfloat16_t, bfloat16_t>;
    template struct GemmResource<float16_t, float16_t>;
    template struct GemmResource<hfloat16_t, hfloat16_t>;
#endif // ROCWMMA_EXTENDED_TESTS

}
