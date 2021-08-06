#ifndef WMMA_H_
#define WMMA_H_

#include <type_traits>

#include "IOTraits.h"
#include "IOUnpack.h"
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

    /**
 * \ingroup wmma
 * \defgroup WMMA APIs
 *
 * @brief WMMA Fragment and its API function definitions.
 */

    /**
 * \ingroup WMMA APIs
 * @{
 */

/*! \class fragment 
 *  \brief Definition of MFMA Fragment
 *
 * LeadingDim - Matrix A loads by col size BlockM
 *              Matrix B / C load by row size BlockN
 * 
 * KDim -  Matrix C loads BlockM rows size BlockN
 *          Matrix A and B load BlockK strides of leading dim.
 * 
 * AccessT - Unpacked data storage
 * 
 * StorageT = Packed data storage accepted by MFMA
 * 
 */
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
            };

        private:
            using IOTraits  = amdgcn_io_traits<LeadingDim, KDim, DataT>;
            using PackedT   = typename PackTraits<DataT>::PackedT;
            using UnpackedT = typename PackTraits<DataT>::UnpackedT;

        public:
            using AccessT  = VecT<UnpackedT, IOTraits::UnpackedSize>;
            using StorageT = VecT<PackedT, IOTraits::PackedSize>;
        };

        __device__ fragment() = default;
        __device__ fragment(const fragment& other);
        __device__ fragment& operator=(const fragment& other);

        // Accessors
        __device__ inline DataT&                           operator[](uint32_t index);
        __device__ inline DataT const&                     operator[](uint32_t index) const;
        __device__ inline typename Traits::StorageT&       operator*();
        __device__ inline typename Traits::StorageT const& operator*() const;

        // Traits
        __device__ constexpr static inline uint32_t leadingDim();
        __device__ constexpr static inline uint32_t kDim();
        __device__ constexpr static inline uint32_t size();

        // Compatibility with nvcuda::wmma
        union
        {
            typename Traits::StorageT mStorage;
            typename Traits::AccessT  mStorageUnpacked;
            typename Traits::AccessT  x;
            static_assert(sizeof(typename Traits::AccessT) == sizeof(typename Traits::StorageT),
                          "Storage type and access type should be views into the same raw data");
        };
        constexpr static uint32_t num_elements = Traits::AccessT::Size;
        using element_type                     = DataT;
    };

    /**
 * \ingroup WMMA APIs
 * @{
 */
    /**
 * 
 * @param frag - Fragment of type MatrixT with its associated block sizes, data type and layout
 * @param value - Value of type DataT
 * 
 * Fills the entire fragment with the desired value 
 */

    template <typename MatrixT,
              uint32_t BlockM,
              uint32_t BlockN,
              uint32_t BlockK,
              typename DataT,
              typename DataLayout>
    __device__ void
        fill_fragment(fragment<MatrixT, BlockM, BlockN, BlockK, DataT, DataLayout>& frag,
                      DataT                                                         value);
    /**
 * \ingroup WMMA APIs
 * @{
 */
    /**
 * 
 * @param frag - Fragment of type MatrixT with its associated block sizes, data type and layout
 * @param data - Data pointer to global/local memory
 * @param ldm - Leading dimension size
 * 
 * 
 * Loads the entire fragment with data from global device memory according to 
 * the matrix layouts
 * 
 */
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

    /**
 * \ingroup WMMA APIs
 * @{
 */
    /**
 * 
 * @param frag - Fragment of type MatrixT with its associated block sizes, data type and layout
 * @param data - Data pointer to global/local memory
 * @param ldm - Leading dimension size
 * @param layout - Matrix layout
 * 
 * 
 * Loads the entire fragment with data from global device memory according to 
 * the matrix layouts
 * 
 */
    template <typename MatrixT, uint32_t BlockM, uint32_t BlockN, uint32_t BlockK, typename DataT>
    __device__ void load_matrix_sync(fragment<MatrixT, BlockM, BlockN, BlockK, DataT>& frag,
                                     const DataT*                                      data,
                                     uint32_t                                          ldm,
                                     layout_t                                          layout);
   /**
 * \ingroup WMMA APIs
 * @{
 */
    /**
 * Cooperative Load Matrix
 * 
 * @param frag - Fragment of type MatrixT with its associated block sizes, data type and layout
 * @param data - Data pointer to global/local memory
 * @param ldm - Leading dimension size
 * 
 * Loads the entire fragment with data from global device memory cooperatively
 * across waves. Each cooperative wave is responsible in loading a piece
 * of the final output.
 * 
 */
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
    /**
 * \ingroup Fragment & APIs
 * @{
 */
    /**
 * 
 * 
 * @param frag - Fragment of type MatrixT with its associated block sizes, data type and layout
 * @param data - Data pointer to global/local memory
 * @param ldm - Leading dimension size
 * 
 * 
 * Stores the entire fragment into a global/local datapointer according to 
 * the matrix layouts
 * 
 */
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

    /**
 * \ingroup Fragment & APIs
 * @{
 */
    /**
 * 
 * 
 * @param frag - Fragment of type MatrixT with its associated block sizes, data type and layout
 * @param data - Data pointer to global/local memory
 * @param ldm - Leading dimension size
 * @param layout - Matrix layout
 * 
 * 
 * Stores the entire fragment into a global/local datapointer according to 
 * the matrix layouts
 * 
 */

    template <typename MatrixT, uint32_t BlockM, uint32_t BlockN, uint32_t BlockK, typename DataT>
    __device__ void store_matrix_sync(DataT*                                                  data,
                                      fragment<MatrixT, BlockM, BlockN, BlockK, DataT> const& frag,
                                      uint32_t                                                ldm,
                                      layout_t layout);

    /**
 * \ingroup WMMA APIs
 * @{
 */
    /**
 * 
 * @param d - Accumulator fragment D
 * @param a - Input fragment A
 * @param b - Input fragment B
 * @param c - Input/Accumulator fragment C
 * 
 * 
 * Performs the Multiply-Accumulate operation on the fragments A, B, C and D(D = A * B + C)
 * 
 */
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

    /**
 * \ingroup WMMA APIs
 * @{
 */
    /**
 * 
 * Performs synchronization across multiple wavefronts in a workgroup.
 * 
 */
    __device__ void wmma_s_barrier();

} // namespace wmma

#include "WMMA_impl.h"

#endif // WMMA_H_
