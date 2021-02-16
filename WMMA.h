#ifndef WMMA_H_
#define WMMA_H_

#include <type_traits>
#include "Types.h"

namespace wmma
{
    template <typename MatrixT,
              uint32_t BlockM,
              uint32_t BlockN,
              uint32_t BlockK,
              typename DataT,
              typename LayoutT = void>
    class __align__(4) fragment
    {
    public:

        struct Traits
        {
            enum : uint32_t
            {
                // Matrix A loads by col size BlockM
                // Matrix B / C load by row size BlockN
                LeadingDim = std::is_same<MatrixT, matrix_a>::value ? BlockM : BlockN,

                // Matrix C loads BlockM rows size BlockN
                // Matrix A and B load BlockK strides of leading dim.
                KDim = std::is_same<MatrixT, accumulator>::value ? BlockM : BlockK,

                ElementCount = (std::is_same<MatrixT, matrix_b>::value ? BlockK : BlockM) *
                               (std::is_same<MatrixT, matrix_a>::value ? BlockK : BlockN),

                RegisterCount = ElementCount * sizeof(DataT) / BYTES_PER_REGISTER,
            };

            static_assert((ElementCount * sizeof(DataT)) % BYTES_PER_REGISTER == 0, "Partial registers unsupported");

            using StorageT = VecT<DataT, RegisterCount>;
        };

        // Accessors
        __device__ inline DataT& operator[](uint32_t index);
        __device__ inline DataT const& operator[](uint32_t index) const;
        __device__ inline typename Traits::StorageT& operator*();
        __device__ inline typename Traits::StorageT const& operator*() const;

        // Traits
        __device__ constexpr static inline uint32_t leadingDim();
        __device__ constexpr static inline uint32_t kDim();
        __device__ constexpr static inline uint32_t elementCount();
        __device__ constexpr static inline uint32_t registerCount();

    private:
        typename Traits::StorageT mStorage;
    };

    template <typename MatrixT, uint32_t BlockM, uint32_t BlockN, uint32_t BlockK, typename DataT, typename DataLayout>
    __device__ void fill_fragment(fragment<MatrixT, BlockM, BlockN, BlockK, DataT, DataLayout>& frag,
                                  DataT                                                      value);

    template <typename MatrixT, uint32_t BlockM, uint32_t BlockN, uint32_t BlockK, typename DataT, typename DataLayout>
    __device__ void load_matrix_sync(fragment<MatrixT, BlockM, BlockN, BlockK, DataT, DataLayout>& frag,
                         const DataT*                                               data,
                         uint32_t                                                   ldm);
    

    template <typename MatrixT, uint32_t BlockM, uint32_t BlockN, uint32_t BlockK, typename DataT>
    __device__ void load_matrix_sync(fragment<MatrixT, BlockM, BlockN, BlockK, DataT>& frag,
                                     const DataT*                                      data,
                                     uint32_t                                          ldm,
                                     layout_t                                          layout);

    template <typename MatrixT, uint32_t BlockM, uint32_t BlockN, uint32_t BlockK, typename DataT, typename DataLayout>
    __device__ void load_matrix_coop_sync(fragment<MatrixT, BlockM, BlockN, BlockK, DataT, DataLayout>& frag,
                         const DataT*                                               data,
                         uint32_t                                                   ldm);

    template <typename MatrixT, uint32_t BlockM, uint32_t BlockN, uint32_t BlockK, typename DataT, typename DataLayout>
    __device__ void store_matrix_sync(DataT*                                                           data,
                          fragment<MatrixT, BlockM, BlockN, BlockK, DataT, DataLayout> const& frag,
                          uint32_t                                                         ldm);

    template <typename MatrixT, uint32_t BlockM, uint32_t BlockN, uint32_t BlockK, typename DataT>
    __device__ void store_matrix_sync(DataT*                                                  data,
                                      fragment<MatrixT, BlockM, BlockN, BlockK, DataT> const& frag,
                                      uint32_t                                                ldm,
                                      layout_t layout);

    template <uint32_t BlockM, uint32_t BlockN, uint32_t BlockK, typename InputT, typename ComputeT, typename LayoutA, typename LayoutB>
    __device__ void mma_sync(fragment<accumulator, BlockM, BlockN, BlockK, ComputeT>&           d,
                             fragment<matrix_a, BlockM, BlockN, BlockK, InputT, LayoutA> const& a,
                             fragment<matrix_b, BlockM, BlockN, BlockK, InputT, LayoutB> const& b,
                             fragment<accumulator, BlockM, BlockN, BlockK, ComputeT> const&     c);

   

} // namespace wmma

#include "WMMA_impl.h"

#endif // WMMA_H_
