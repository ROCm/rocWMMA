#ifndef WMMA_H_
#define WMMA_H_

#include <type_traits>

#include "IOTraits.h"
#include "IOUnpack.h"
#include "Types.h"

namespace wmma
{
    // Meta-tags

    // Matrices
    using row_major   = ::row_major;
    using col_major   = ::col_major;
    using matrix_a    = ::matrix_a;
    using matrix_b    = ::matrix_b;
    using accumulator = ::accumulator;
    using common      = ::common;

    // Memory
    using globalMem = ::globalMem;
    using ldsMem    = ::ldsMem;

    enum layout_t : uint32_t
    {
        mem_row_major,
        mem_col_major
    };

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

                ElementCount = (std::is_same<MatrixT, matrix_b>::value ? BlockK : BlockM)
                               * (std::is_same<MatrixT, matrix_a>::value ? BlockK : BlockN),

                // Packed elements
                RegisterCount = ElementCount * sizeof(DataT) / BYTES_PER_REGISTER,
            };

            static_assert((ElementCount * sizeof(DataT)) % BYTES_PER_REGISTER == 0,
                          "Partial registers unsupported");

            using PackedT  = typename PackTraits<DataT>::PackedT;
            using AccessT  = typename Unpack<DataT, RegisterCount>::Traits::OutputT;
            using StorageT = VecT<PackedT, RegisterCount>;
        };

        __device__ fragment() = default;
        __device__ fragment(const fragment& other);
        __device__ fragment& operator=(const fragment& other);

        // Accessors
        __device__ inline DataT&                           operator[](uint32_t index);
        __device__ inline DataT const&                     operator[](uint32_t index) const;
        __device__ inline typename Traits::StorageT&       operator*();
        __device__ inline typename Traits::StorageT const& operator*() const;
        __device__ inline uint32_t                         size();

        // Traits
        __device__ constexpr static inline uint32_t leadingDim();
        __device__ constexpr static inline uint32_t kDim();
        __device__ constexpr static inline uint32_t elementCount();
        __device__ constexpr static inline uint32_t registerCount();

        // Compatibility with nvcuda::wmma
        union
        {
            typename Traits::StorageT mStorage;
            typename Traits::AccessT  mStorageUnpacked;
            typename Traits::AccessT  x;
            static_assert(sizeof(typename Traits::AccessT) == sizeof(typename Traits::StorageT),
                          "Storage type and access type should be views into the same raw data");
        };
        static const uint32_t num_elements = Traits::AccessT::Size;
        using element_type                 = DataT;
    };

    template <typename MatrixT,
              uint32_t BlockM,
              uint32_t BlockN,
              uint32_t BlockK,
              typename DataT,
              typename DataLayout>
    __device__ void
        fill_fragment(fragment<MatrixT, BlockM, BlockN, BlockK, DataT, DataLayout>& frag,
                      DataT                                                         value);

    template <typename MatrixT,
              uint32_t BlockM,
              uint32_t BlockN,
              uint32_t BlockK,
              typename DataT,
              typename DataLayout>
    __device__ void
        load_matrix_sync(fragment<MatrixT, BlockM, BlockN, BlockK, DataT, DataLayout>& frag,
                         const DataT*                                                  data,
                         uint32_t                                                      ldm);

    template <typename MatrixT, uint32_t BlockM, uint32_t BlockN, uint32_t BlockK, typename DataT>
    __device__ void load_matrix_sync(fragment<MatrixT, BlockM, BlockN, BlockK, DataT>& frag,
                                     const DataT*                                      data,
                                     uint32_t                                          ldm,
                                     layout_t                                          layout);

    template <typename MatrixT,
              uint32_t BlockM,
              uint32_t BlockN,
              uint32_t BlockK,
              typename DataT,
              typename DataLayout>
    __device__ void
        load_matrix_coop_sync(fragment<MatrixT, BlockM, BlockN, BlockK, DataT, DataLayout>& frag,
                              const DataT*                                                  data,
                              uint32_t                                                      ldm);

    template <typename MatrixT,
              uint32_t BlockM,
              uint32_t BlockN,
              uint32_t BlockK,
              typename DataT,
              typename DataLayout>
    __device__ void
        store_matrix_sync(DataT*                                                              data,
                          fragment<MatrixT, BlockM, BlockN, BlockK, DataT, DataLayout> const& frag,
                          uint32_t                                                            ldm);

    template <typename MatrixT, uint32_t BlockM, uint32_t BlockN, uint32_t BlockK, typename DataT>
    __device__ void store_matrix_sync(DataT*                                                  data,
                                      fragment<MatrixT, BlockM, BlockN, BlockK, DataT> const& frag,
                                      uint32_t                                                ldm,
                                      layout_t layout);

    template <uint32_t BlockM,
              uint32_t BlockN,
              uint32_t BlockK,
              typename InputT,
              typename ComputeT,
              typename LayoutA,
              typename LayoutB>
    __device__ void mma_sync(fragment<accumulator, BlockM, BlockN, BlockK, ComputeT>&           d,
                             fragment<matrix_a, BlockM, BlockN, BlockK, InputT, LayoutA> const& a,
                             fragment<matrix_b, BlockM, BlockN, BlockK, InputT, LayoutB> const& b,
                             fragment<accumulator, BlockM, BlockN, BlockK, ComputeT> const&     c);

} // namespace wmma

#include "WMMA_impl.h"

#endif // WMMA_H_
