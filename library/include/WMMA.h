#ifndef WMMA_H_
#define WMMA_H_

#include <type_traits>

#include "IOTraits.h"
#include "Types.h"

/**
 * \mainpage
 * 
 * WMMA is a C++ library for facilitating GEMM, or GEMM-like 2D matrix multiplications 
 * leveraging AMD's GPU hardware matrix cores through HIP.
 * Specifically, the library enhances the portability of CUDA WMMA code to 
 * AMD's heterogeneous platform and provides an interface to use underlying 
 * hardware matrix multiplication (MFMA) units.
 * The WMMA API exposes memory and MMA (Matrix Multiply Accumulate) functions
 * that operate on blocks, or 'fragments' of data appropriately sized for 
 * warp (thread block) execution.
 * WMMA code is templated for componentization and for providing ability to 
 * make compile-time optimizations based on available meta-data.
 * This library is an ongoing Work-In-Progress (WIP).
 * 
 * 
 * **Supported Datatypes**
 *  - Native Data Types
 *      - float = f32
 *      - double = f64
 *      - _Float16 = f16
 *      - int8
 *      - uint8
 *      - int16
 *      - int32
 *      - uint32
 * 
 * 
 *  - Non-Native Data Types
 *      - h16 = __half
 *      - bf16 = bfloat16
 * 
 * **Supported Thread Block Sizes**
 * 
 * TBlockX  | TBlockY   |
 * :-------:|:---------:|
 * 64       |   1       |
 * 64       |   2       | 
 * 64       |   4       |
 * 64       |   8       | 
 * 128      |   1       |
 * 128      |   2       | 
 * 256      |   1       |
 * 
 * @note TBlockX must be a multiple of 64
 * 
 * 
 * **Supported Matrix Layouts**
 * 
 * Matrix Layout(N = col major, T = row major) 
 * 
 * LayoutA  |   LayoutB |   LayoutC |   LayoutD  |
 * :-------:|:---------:|:---------:|:----------:|
 *     N    |      N    |      N    |     N      |
 *     N    |      T    |      N    |     N      |
 *     T    |      N    |      N    |     N      |
 *     T    |      T    |      N    |     N      |
 *     N    |      N    |      T    |     T      |
 *     N    |      T    |      T    |     T      |
 *     T    |      N    |      T    |     T      |
 *     T    |      T    |      T    |     T      |
 * 
 * 
 * 
 * **Data Types <Ti / To / Tc> = <InputType / OutputType / ComputeType >**
 * \n
 * **MFMA Block Size = <BlockM, BlockN, BlockK>**
 * \n
 * Ti / To / Tc         |   BlockM    |   BlockN    |   BlockK
 * :-------------------:|:-----------:|:-----------:|:-----------:|
 * i8/i32/i32           |    16       |      16     | Min:16,pow2 |
 * ^                    |    32       |      32     | Min:8, pow2 |
 * i8/i8/i32            |    16       |      16     | Min:16,pow2 |           
 * ^                    |    32       |      32     | Min:8, pow2 |
 * f16/f32/f32          |    16       |      16     | Min:16,pow2 |
 * ^                    |    32       |      32     | Min:8, pow2 |
 * f16/f16/f32          |    16       |      16     | Min:16,pow2 |
 * ^                    |    32       |      32     | Min:8, pow2 |
 * f16/f16/f16          |    16       |      16     | Min:16,pow2 |
 * ^                    |    32       |      32     | Min:8, pow2 |
 * __half/f32/f32       |    16       |      16     | Min:16,pow2 |
 * ^                    |    32       |      32     | Min:8, pow2 |
 * __half/__half/f32    |    16       |      16     | Min:16,pow2 |
 * ^                    |    32       |      32     | Min:8, pow2 |
 * __half/__half/__half |    16       |      16     | Min:16,pow2 |
 * ^                    |    32       |      32     | Min:8, pow2 |
 * bf16/f32/f32         |    16       |      16     | Min:8, pow2 |
 * ^                    |    32       |      32     | Min:4, pow2 |
 * bf16/bf16/f32        |    16       |      16     | Min:8, pow2 |
 * ^                    |    32       |      32     | Min:4, pow2 |
 * bf16/bf16/bf16       |    16       |      16     | Min:8, pow2 |
 * ^                    |    32       |      32     | Min:4, pow2 |
 * f32/f32/f32          |    16       |      16     | Min:4, pow2 |
 * ^                    |    32       |      32     | Min:2, pow2 |
 * f64/f64/f64          |    16       |      16     | Min:4, pow2 |
 *   
 * 
 * \n
 * 
 * **LDS Support/Requirements(Only applicable to WMMA v0.3 and below)**
 * 
 * Required LDS space is calculated by
 *  - LDSBytes = max(BlockM * blockDim.y, BlockN * blockDim.x / 64) * BlockK * sizeof(InputT)
 * 
 * 
 * \n
 * **Fragment:**
 * 
 * **fill_fragment**
 * 
 * Broadcast a desired value to all elements in the fragment.
 * 
 * \n
 * **load_matrix_sync / store_matrix_sync**
 * 
 * Loads data from memory according to Matrix Layout.
 * Matrix A layout loads / stores matrix columns in the K direction 
 * (Matrix A = M x K, fragA = BlockM x BlockK)
 * Matrix B layout loads / stores matrix rows in the K direction 
 * (Matrix B = K x N, fragB = BlockK x BlockN)
 * Matrix C layout loads / stores matrix rows in vector width of 4 
 * (Matrix C = M x N, fragAcc = BlockM x BlockN)
 * 
 * Fragments are stored in packed registers in optimal load / store patterns. In-register elements have no guaranteed order, which have been optimized for loading / storing efficiency.
 * 
 * \n
 * **mma_sync**
 * 
 * MFMA accumulation is performed with fragment data. Fragment A cols are multiplied
 * with Fragment B rows and added to the accumulator fragment.
 */

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

            using PackedT = typename PackTraits<DataT>::PackedT;

            using StorageT = VecT<PackedT, RegisterCount>;
        };

        // Accessors
        __device__ inline DataT&                           operator[](uint32_t index);
        __device__ inline DataT const&                     operator[](uint32_t index) const;
        __device__ inline typename Traits::StorageT&       operator*();
        __device__ inline typename Traits::StorageT const& operator*() const;

        // Traits
        __device__ constexpr static inline uint32_t leadingDim();
        __device__ constexpr static inline uint32_t kDim();
        __device__ constexpr static inline uint32_t elementCount();
        __device__ constexpr static inline uint32_t registerCount();

    private:
        typename Traits::StorageT mStorage;
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
