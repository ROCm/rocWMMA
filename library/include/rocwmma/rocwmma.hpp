/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2021-2025 Advanced Micro Devices, Inc. All rights reserved.
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
#ifndef ROCWMMA_API_HPP
#define ROCWMMA_API_HPP

#include "internal/accessors.hpp"
#include "internal/io_traits.hpp"
#include "internal/pack_util.hpp"
#include "internal/types.hpp"

/**
 * \mainpage
 *
 * rocWMMA is a C++ header library for accelerating mixed precision matrix multiply-accumulate operations
 * leveraging specialized GPU matrix cores on AMD's latest discrete GPUs. 'roc' being an AMD-specific
 * component belonging to the ROCm ecosystem, and WMMA stands for Wavefront Mixed precision Multiply Accumulate.
 *
 * rocWMMA leverages modern C++ techniques. It is templated for modularity and uses meta-programming paradigms to provide opportunities for customization
 * and compile-time inferences and optimizations. The API is seamless across supported CDNA and RDNA architectures. It is also portable with the Nvidia
 * nvcuda::wmma library, allowing those users to easily migrate to the AMD platform.
 *
 * The API is implemented as GPU device code which empowers users with direct use of GPU matrix cores, right from their kernel code.
 * Major benefits include kernel-level control which allows authoring flexibility and accessibility to compiler optimization passes in-situ
 * with other device code. Users can therefore decide when and where kernel run-time launches are required, which is not dictated by the API.
 *
 * rocWMMA's API facilitates the decomposition of matrix multiply-accumulate problems into discretized blocks (also known as fragments) and enables
 * parallelization of block-wise operations across multiple GPU wavefronts. The programmer's perspective is simplified to wavefront handling of fragments,
 * whereas individual threads are handled internally. This can allow for faster development times and a more seamless experience across multiple architectures.
 * API functions include data loading and storing, matrix multiply-accumulate and helper transforms that operate on data fragment abstractions. Moreover, data movement
 * between global and local memory can be done cooperatively amongst the wavefronts in a threadblock to enable data sharing and re-use. Matrix multiply-accumulate
 * functionality supports mixed precision inputs and outputs with native fixed-precision accumulation.
 *
 * Supporting code is required for GPU device management and kernel invocation. The kernel code samples and tests provided are built and launched via
 * the Heterogeneous-Compute Interface for Portability (HIP) ecosystem within ROCm.
 *
 * This library is an ongoing Work-In-Progress (WIP).
 *
 * For more documentation, please visit https://rocm.docs.amd.com/projects/rocWMMA/en/latest/index.html.
 *
*/

namespace rocwmma
{
    //! @defgroup Rocwmma rocWMMA Public API
    //!
    //! @brief rocWMMA objects and API function definitions.
    //! @{

    //! @struct row_major
    //! @brief Meta-tag indicating 2D in-memory data layout as row major.
    struct row_major
    {
    };

    //! @struct col_major
    //! @brief Meta-tag indicating 2D in-memory data layout as column major.
    struct col_major
    {
    };

    //! @struct matrix_a
    //! @brief Meta-tag indicating data context is input Matrix A.
    struct matrix_a
    {
    };

    //! @struct matrix_b
    //! @brief Meta-tag indicating data context is input Matrix B.
    struct matrix_b
    {
    };

    //! @struct accumulator
    //! @brief Meta-tag indicating data context is Accumulator (also used as Matrix C / D).
    struct accumulator
    {
    };

    namespace fragment_schedule
    {
        // No thread-block cooperation
        struct non_cooperative
        {
            constexpr static inline auto waveIndex()
            {
                return 0u;
            }
            constexpr static inline uint32_t waveCount()
            {
                return 1u;
            }
        };

        // Thread-block schedule is round-robin in row_major; all waves participate.
        // E.g. (TBlockX, TBlockY) = (128, 2) = 2x2 waves
        // i0 = (0, 0), i1 = (0, 1),
        // i2 = (1, 0), i3 = (1, 1)
        // count = 4
        template <uint32_t TBlockX = 0, uint32_t TBlockY = 0>
        struct coop_rr_row_major
        {
            using WaveSpace = detail::WaveSpace<TBlockX, TBlockY>;
            using DataSpace = detail::DataSpace<row_major>;

            constexpr static inline auto waveIndex()
            {
                return DataSpace::fromMatrixCoord(WaveSpace::localWaveCoord(),
                                                  get<1>(WaveSpace::workgroupDim()));
            }
            constexpr static inline uint32_t waveCount()
            {
                return reduce_mult(WaveSpace::workgroupDim());
            }
        };

        // Thread-block schedule is round-robin in col_major; all waves participate.
        // E.g. (TBlockX, TBlockY) = (128, 2) = 2x2 waves
        // i0 = (0, 0), i2 = (0, 1),
        // i1 = (1, 0), i3 = (1, 1),
        // count = 4
        template <uint32_t TBlockX = 0, uint32_t TBlockY = 0>
        struct coop_rr_col_major
        {
            using WaveSpace = detail::WaveSpace<TBlockX, TBlockY>;
            using DataSpace = detail::DataSpace<col_major>;

            constexpr static inline auto waveIndex()
            {
                return DataSpace::fromMatrixCoord(WaveSpace::localWaveCoord(),
                                                  get<0>(WaveSpace::workgroupDim()));
            }
            constexpr static inline uint32_t waveCount()
            {
                return reduce_mult(WaveSpace::workgroupDim());
            }
        };

        // Thread-block schedule is sliced into rows; all waves in the same row participate.
        // E.g. Wg = (128, 2) = 2x2 waves
        // Slice0: i0 = (0, 0), i1 = (0, 1) count = 2
        // Slice1: i0 = (1, 0), i1 = (1, 1) count = 2
        template <uint32_t TBlockX = 0, uint32_t TBlockY = 0>
        struct coop_slice_row
        {
            using WaveSpace = detail::WaveSpace<TBlockX, TBlockY>;

            constexpr static inline auto waveIndex()
            {
                return get<1>(WaveSpace::localWaveCoord());
            }
            constexpr static inline uint32_t waveCount()
            {
                return get<1>(WaveSpace::workgroupDim());
            }
        };

        // Thread-block schedule is sliced into cols; all waves in the same col participate.
        // E.g. Wg = (128, 2) = 2x2 waves
        // Slice0:        Slice1:
        // i0 = (0, 0),   i0 = (0, 1),
        // i1 = (1, 0)    i1 = (1, 1)
        // count = 2      count = 2
        template <uint32_t TBlockX = 0, uint32_t TBlockY = 0>
        struct coop_slice_col
        {
            using WaveSpace = detail::WaveSpace<TBlockX, TBlockY>;

            constexpr static inline auto waveIndex()
            {
                return get<0>(WaveSpace::localWaveCoord());
            }
            constexpr static inline uint32_t waveCount()
            {
                return get<0>(WaveSpace::workgroupDim());
            }
        };

    } // namespace fragment_schedule

    namespace ScheduleTraits_impl
    {
        using namespace fragment_schedule;

        template <typename ScheduleT>
        struct is_schedule_valid : false_type
        {
        };

        template <>
        struct is_schedule_valid<non_cooperative> : true_type
        {
        };

        template <uint32_t TBlockX, uint32_t TBlockY>
        struct is_schedule_valid<coop_rr_row_major<TBlockX, TBlockY>> : true_type
        {
        };

        template <uint32_t TBlockX, uint32_t TBlockY>
        struct is_schedule_valid<coop_rr_col_major<TBlockX, TBlockY>> : true_type
        {
        };

        template <uint32_t TBlockX, uint32_t TBlockY>
        struct is_schedule_valid<coop_slice_row<TBlockX, TBlockY>> : true_type
        {
        };

        template <uint32_t TBlockX, uint32_t TBlockY>
        struct is_schedule_valid<coop_slice_col<TBlockX, TBlockY>> : true_type
        {
        };

        template <typename ScheduleT>
        struct is_schedule_constexpr : false_type
        {
        };

        template <uint32_t TBlockX, uint32_t TBlockY, template <uint32_t, uint32_t> class Schedule>
        struct is_schedule_constexpr<Schedule<TBlockX, TBlockY>>
            : integral_constant<bool, (TBlockX > 0u && TBlockY > 0u)>
        {
        };

        template <typename ScheduleT>
        struct is_schedule_cooperative : false_type
        {
        };

        template <uint32_t TBlockX, uint32_t TBlockY>
        struct is_schedule_cooperative<coop_rr_row_major<TBlockX, TBlockY>> : true_type
        {
        };

        template <uint32_t TBlockX, uint32_t TBlockY>
        struct is_schedule_cooperative<coop_rr_col_major<TBlockX, TBlockY>> : true_type
        {
        };

        template <uint32_t TBlockX, uint32_t TBlockY>
        struct is_schedule_cooperative<coop_slice_row<TBlockX, TBlockY>> : true_type
        {
        };

        template <uint32_t TBlockX, uint32_t TBlockY>
        struct is_schedule_cooperative<coop_slice_col<TBlockX, TBlockY>> : true_type
        {
        };

    } // namespace ScheduleTraits_impl

    template <typename ScheduleT>
    struct schedule_traits
    {
        constexpr static bool is_schedule_constexpr
            = ScheduleTraits_impl::is_schedule_constexpr<ScheduleT>::value;
        constexpr static bool is_schedule_valid
            = ScheduleTraits_impl::is_schedule_valid<ScheduleT>::value;
        constexpr static bool is_schedule_cooperative
            = ScheduleTraits_impl::is_schedule_cooperative<ScheduleT>::value;
    };

    //! @struct layout_t
    //! @brief Runtime data layout tags
    //! @var mem_row_major
    //! @var mem_col_major
    enum layout_t : uint32_t
    {
        mem_row_major,
        mem_col_major
    };

    //! @class fragment
    //! @brief rocWMMA fragment class. This is the primary object used in block-wise decomposition of the matrix multiply-accumulate (mma)
    //! problem space. In general, fragment data is associated with a matrix context (matrix_a, matrix_b or accumulator), a block size (BlockM/N/K),
    //! a datatype (e.g. single-precision float, etc.) and an in-memory 2D layout (e.g. row_major or col_major). These fragment properties are used
    //! to define how data is handled and stored locally, and to drive API implementations for loading / storing, mma and transforms. Fragment abstractions are
    //! designed to promote a simple wavefront programming model, which can accelerate development time. Internal thread-level details are handled by rocWMMA
    //! which frees the user to focus on wavefront block-wise decomposition. Written purely in device code, the programmer can use this object in their own
    //! device kernels.
    //!
    //! @tparam MatrixT fragment context
    //! @tparam BlockM/N/K block dimensions
    //! @tparam DataT datatype
    //! @tparam DataLayoutT in-memory layout as col_major or row_major
    //!
    //! @note Fragments are stored in packed registers, however vector elements have no guaranteed order or locality.
    template <typename MatrixT,
              uint32_t FragM,
              uint32_t FragN,
              uint32_t FragK,
              typename DataT,
              typename DataLayoutT = void,
              typename ScheduleT   = fragment_schedule::non_cooperative>
    class __align__(4) fragment
    {
    public:
        //! Input / output traits specific to AMDGCN architecture
        using IOTraits =
            typename IOConfig<MatrixT, FragM, FragN, FragK, DataT, DataLayoutT>::IOTraits;

        struct Traits
        {
        private:
            //! The packed type for element data
            using PackedElementT = typename PackTraits<DataT>::PackedT;

            //! The unpacked type for element data
            using UnpackedElementT = typename PackTraits<DataT>::UnpackedT;

        public:
            //! Unpacked data access view
            using AccessT = VecT<UnpackedElementT, IOTraits::UnpackedSize>;

            //! Packed data storage view
            using StorageT = VecT<PackedElementT, IOTraits::PackedSize>;

            constexpr static uint32_t Size = IOTraits::UnpackedSize;

            static_assert(IOTraits::PackedVRegCount >= 1,
                          "Fragments must occupy at least one packed register");
            static_assert(IOTraits::UnpackedSize % IOTraits::PackedSize == 0,
                          "Unable to pack fragment elements");
        };

        ROCWMMA_DEVICE           fragment() = default;
        ROCWMMA_DEVICE           fragment(const fragment& other);
        ROCWMMA_DEVICE fragment& operator=(const fragment& other);

        //! @param index Element index
        //! @returns Mutable unpacked element accessor at given index
        ROCWMMA_DEVICE inline DataT& operator[](uint32_t index);
        //! @param index Element index
        //! @returns Immutable unpacked element accessor at given index
        ROCWMMA_DEVICE inline DataT const& operator[](uint32_t index) const;
        //! @returns Mutable packed storage vector accessor
        ROCWMMA_DEVICE inline typename Traits::StorageT& operator*();
        //! @returns Immutable packed storage vector accessor
        ROCWMMA_DEVICE inline typename Traits::StorageT const& operator*() const;

        //! @returns The geometric height of fragment
        ROCWMMA_DEVICE constexpr static inline uint32_t height();
        //! @returns The geometric width of fragment
        ROCWMMA_DEVICE constexpr static inline uint32_t width();
        //! @returns The leading block dimension (non-K)
        ROCWMMA_DEVICE constexpr static inline uint32_t blockDim();
        //! @returns The k dimension
        ROCWMMA_DEVICE constexpr static inline uint32_t kDim();
        //! @returns The size of the unpacked elements vector
        ROCWMMA_DEVICE constexpr static inline uint32_t size();

        //! Internal data storage views. Compatibility with nvcuda::wmma
        union
        {
            typename Traits::StorageT             mStorage; // Packed
            typename Traits::AccessT              mAccess; // Unpacked
            typename Traits::AccessT::Native_vec_ x; // Nuanced access
            static_assert(sizeof(typename Traits::AccessT) == sizeof(typename Traits::StorageT),
                          "Storage type and access type should be views into the same raw data");
        };

        // For compatibility
        constexpr static uint32_t num_elements = Traits::Size;
        using element_type                     = DataT;
    };

    namespace fragment_traits_impl
    {
        using LayoutTraits_impl::is_col_major;
        using LayoutTraits_impl::is_row_major;
        template <typename MatrixT>
        struct is_matrix_a : false_type
        {
        };

        template <>
        struct is_matrix_a<matrix_a> : true_type
        {
        };

        template <typename MatrixT>
        struct is_matrix_b : false_type
        {
        };

        template <>
        struct is_matrix_b<matrix_b> : true_type
        {
        };

        template <typename MatrixT>
        struct is_accumulator : false_type
        {
        };

        template <>
        struct is_accumulator<accumulator> : true_type
        {
        };

        template <typename FragT>
        struct fragment_traits;

    } // namespace fragment_traits_impl

    template <typename _MatrixT,
              uint32_t _FragM,
              uint32_t _FragN,
              uint32_t _FragK,
              typename _DataT,
              typename _DataLayoutT,
              typename _ScheduleT>
    struct fragment_traits<
        fragment<_MatrixT, _FragM, _FragN, _FragK, _DataT, _DataLayoutT, _ScheduleT>>
        : public fragment<_MatrixT, _FragM, _FragN, _FragK, _DataT, _DataLayoutT, _ScheduleT>::
              Traits,
          public schedule_traits<_ScheduleT>
    {
        using MatrixT                   = _MatrixT;
        constexpr static uint32_t FragM = _FragM;
        constexpr static uint32_t FragN = _FragN;
        constexpr static uint32_t FragK = _FragK;
        using DataT                     = _DataT;
        using DataLayoutT               = _DataLayoutT;
        using ScheduleT                 = _ScheduleT;

        constexpr static bool is_matrix_a    = fragment_traits_impl::is_matrix_a<MatrixT>::value;
        constexpr static bool is_matrix_b    = fragment_traits_impl::is_matrix_b<MatrixT>::value;
        constexpr static bool is_accumulator = fragment_traits_impl::is_accumulator<MatrixT>::value;
        constexpr static bool is_col_major = fragment_traits_impl::is_col_major<DataLayoutT>::value;
        constexpr static bool is_row_major = fragment_traits_impl::is_row_major<DataLayoutT>::value;
    };

    //! Fills the entire fragment with the desired value.
    //! @param frag Fragment of type MatrixT with its associated block sizes, data type and layout
    //! @param value Fill value of type DataT
    //! @tparam Matrix Fragment context
    //! @tparam BlockM/N/K Block dimensions
    //! @tparam DataT Datatype
    //! @tparam DataLayoutT in-memory layout as col_major or row_major
    template <typename FragT, typename DataT>
    ROCWMMA_DEVICE void fill_fragment(FragT& frag, DataT value);

    //! Loads the entire fragment from the data pointer according to its matrix and data layout contexts. Data pointer may point to either local or global memory.
    //! @param frag Fragment of type MatrixT with its associated block sizes, data type and layout
    //! @param data Data pointer to global or local memory
    //! @param ldm Leading dimension size
    //! @tparam MatrixT Fragment context
    //! @tparam BlockM/N/K Block dimensions
    //! @tparam DataT Datatype
    //! @tparam DataLayoutT In-memory layout as col_major or row_major
    template <typename FragT, typename DataT>
    ROCWMMA_DEVICE void load_matrix_sync(FragT& frag, const DataT* data, uint32_t ldm);

    //! Loads the entire fragment from the data pointer according to its matrix layout and data layout contexts.
    //! Data pointer may point to either local or global memory. This overload provides a run-time ability to choose the data layout of the target fragment.
    //! @param frag Fragment of type MatrixT with its associated block sizes, data type and layout
    //! @param data Data pointer to global/local memory
    //! @param ldm Leading dimension size
    //! @param layout Data layout
    //! @tparam MatrixT Fragment context
    //! @tparam BlockM/N/K Block dimensions
    //! @tparam DataT Datatype
    template <typename FragT, typename DataT>
    ROCWMMA_DEVICE void
        load_matrix_sync(FragT& frag, const DataT* data, uint32_t ldm, layout_t layout);

    //! Stores the entire fragment to the data pointer according to its matrix and data layouts. Data pointer may point to either local or global memory.
    //! @param frag Fragment of type MatrixT with its associated block sizes, data type and layout
    //! @param data Data pointer to global/local memory
    //! @param ldm Leading dimension size
    //! @tparam MatrixT Fragment context
    //! @tparam BlockM/N/K Block dimensions
    //! @tparam DataT Datatype
    //! @tparam DataLayoutT in-memory layout as col_major or row_major
    template <typename FragT, typename DataT>
    ROCWMMA_DEVICE void store_matrix_sync(DataT* data, FragT const& frag, uint32_t ldm);

    //! Stores the entire fragment to the data pointer according to its matrix layout. Data pointer may point to either local or global memory.
    //! This overload provides a run-time ability to choose the data layout of the target fragment.
    //! @param frag Fragment of type MatrixT with its associated block sizes, data type and layout
    //! @param data Data pointer to global/local memory
    //! @param ldm Leading dimension size
    //! @param layout Data layout
    //! @tparam MatrixT Fragment context
    //! @tparam BlockM/N/K Block dimensions
    //! @tparam DataT Datatype
    template <typename FragT, typename DataT>
    ROCWMMA_DEVICE void
        store_matrix_sync(DataT* data, FragT const& frag, uint32_t ldm, layout_t layout);

    //! Performs the Multiply-Accumulate operation on the fragments A, B, C and D (D = A * B + C)
    //! @param d Accumulator output D
    //! @param a Input fragment A
    //! @param b Input fragment B
    //! @param c Input accumulator fragment C
    //! @tparam BlockM/N/K block dimensions
    //! @tparam InputT A/B Datatype of input frags A and B
    //! @tparam ComputeT Datatype of accumulator fragment C / D
    //! @tparam LayoutA/B/C/D In-memory layout of frag as col_major or row_major
    //! @note Frag c = d is valid
    template <typename FragA, typename FragB, typename FragAccumIn, typename FragAccumOut>
    ROCWMMA_DEVICE void mma_sync(FragAccumOut& d, FragA const& a, FragB const& b, FragAccumIn& c);

    //! Synchronization point for all wavefronts in a workgroup. Guarantees pending reads / writes to LDS are flushed.
    ROCWMMA_DEVICE ROCWMMA_INLINE void synchronize_workgroup();

    /** @}*/
} // namespace rocwmma

#include "rocwmma_impl.hpp"

#endif // ROCWMMA_API_HPP
