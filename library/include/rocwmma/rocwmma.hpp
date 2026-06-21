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
#include "internal/io_scheduler.hpp"
#include "internal/io_traits.hpp"
#include "internal/pack_util.hpp"
#include "internal/types.hpp"

/**
 * \~english
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
 * \~chinese
 * \mainpage
 *
 * rocWMMA是一个C++头文件库，利用AMD最新独立GPU上的专用GPU矩阵核心，加速混合精度矩阵乘累加运算。
 * 'roc'是属于ROCm生态系统的AMD特定组件，WMMA代表波前混合精度乘累加（Wavefront Mixed precision Multiply Accumulate）。
 *
 * rocWMMA利用现代C++技术。它采用模板化设计以实现模块化，并使用元编程范式提供定制机会以及编译时推断和优化。
 * API在支持的CDNA和RDNA架构上无缝运行。它还与Nvidia的nvcuda::wmma库兼容，使用户能够轻松迁移到AMD平台。
 *
 * API实现为GPU设备代码，使用户能够直接从其内核代码中使用GPU矩阵核心。
 * 主要优势包括内核级控制，允许编写灵活性和访问与其他设备代码原位的编译器优化过程。
 * 因此，用户可以决定何时何地需要内核运行时启动，而不是由API决定。
 *
 * rocWMMA的API促进将矩阵乘累加问题分解为离散块（也称为片段），并实现跨多个GPU波前的块级操作并行化。
 * 程序员的视角简化为波前处理片段，而各个线程在内部处理。这可以加快开发时间，并在多个架构上提供更无缝的体验。
 * API函数包括数据加载和存储、矩阵乘累加以及对数据片段抽象进行操作的辅助转换。此外，全局和本地内存之间的数据移动
 * 可以在线程块中的波前之间协作完成，以实现数据共享和重用。矩阵乘累加功能支持混合精度输入和输出，具有本机固定精度累加。
 *
 * GPU设备管理和内核调用需要支持代码。提供的内核代码示例和测试通过ROCm中的异构计算可移植性接口（HIP）生态系统构建和启动。
 *
 * 此库正在持续开发中（WIP）。
 *
 * 有关更多文档，请访问 https://rocm.docs.amd.com/projects/rocWMMA/en/latest/index.html。
 *
 * \~japanese
 * \mainpage
 *
 * rocWMMAは、AMDの最新ディスクリートGPUの専用GPUマトリックスコアを活用して、混合精度行列乗算累積演算を高速化するC++ヘッダーライブラリです。
 * 'roc'はROCmエコシステムに属するAMD固有のコンポーネントであり、WMMAはWavefront Mixed precision Multiply Accumulate（ウェーブフロント混合精度乗算累積）の略です。
 *
 * rocWMMAは最新のC++技術を活用しています。モジュール性のためにテンプレート化されており、カスタマイズの機会とコンパイル時の推論および最適化を提供するために
 * メタプログラミングパラダイムを使用しています。APIは、サポートされているCDNAおよびRDNAアーキテクチャ全体でシームレスです。また、Nvidiaの
 * nvcuda::wmmaライブラリと互換性があり、ユーザーがAMDプラットフォームに簡単に移行できるようになっています。
 *
 * APIはGPUデバイスコードとして実装されており、ユーザーはカーネルコードから直接GPUマトリックスコアを使用できます。
 * 主な利点には、カーネルレベルの制御が含まれ、これにより、柔軟な作成と他のデバイスコードとのインサイトでコンパイラ最適化パスへのアクセスが可能になります。
 * したがって、ユーザーはカーネルランタイムの起動が必要な時期と場所を決定でき、APIによって指定されることはありません。
 *
 * rocWMMAのAPIは、行列乗算累積問題を離散化されたブロック（フラグメントとも呼ばれる）に分解することを容易にし、
 * 複数のGPUウェーブフロントにわたるブロック単位の演算の並列化を可能にします。プログラマーの視点は、フラグメントのウェーブフロント処理に簡素化され、
 * 個々のスレッドは内部で処理されます。これにより、開発時間を短縮し、複数のアーキテクチャでよりシームレスなエクスペリエンスを提供できます。
 * API関数には、データの読み込みと保存、行列乗算累積、およびデータフラグメント抽象化で動作するヘルパー変換が含まれます。さらに、グローバルメモリと
 * ローカルメモリ間のデータ移動は、スレッドブロック内のウェーブフロント間で協力的に実行して、データの共有と再利用を可能にすることができます。
 * 行列乗算累積機能は、ネイティブ固定精度累積による混合精度入力と出力をサポートします。
 *
 * GPUデバイス管理とカーネル呼び出しにはサポートコードが必要です。提供されるカーネルコードのサンプルとテストは、
 * ROCm内のHeterogeneous-Compute Interface for Portability（HIP）エコシステムを介してビルドおよび起動されます。
 *
 * このライブラリは進行中の作業（WIP）です。
 *
 * 詳細なドキュメントについては、https://rocm.docs.amd.com/projects/rocWMMA/en/latest/index.html をご覧ください。
 *
 * \~
*/

namespace rocwmma
{
    /**
     * \~english
     * @defgroup Rocwmma rocWMMA Public API
     *
     * @brief rocWMMA objects and API function definitions.
     * @{
     *
     * \~chinese
     * @defgroup Rocwmma rocWMMA公共API
     *
     * @brief rocWMMA对象和API函数定义。
     * @{
     *
     * \~japanese
     * @defgroup Rocwmma rocWMMAパブリックAPI
     *
     * @brief rocWMMAオブジェクトとAPI関数の定義。
     * @{
     *
     * \~
     */

    /**
     * \~english
     * @struct row_major
     * @brief Meta-tag indicating 2D in-memory data layout as row major.
     *
     * \~chinese
     * @struct row_major
     * @brief 元标记，表示2D内存数据布局为行主序。
     *
     * \~japanese
     * @struct row_major
     * @brief 2Dメモリ内データレイアウトが行優先であることを示すメタタグ。
     *
     * \~
     */
    struct row_major
    {
    };

    /**
     * \~english
     * @struct col_major
     * @brief Meta-tag indicating 2D in-memory data layout as column major.
     *
     * \~chinese
     * @struct col_major
     * @brief 元标记，表示2D内存数据布局为列主序。
     *
     * \~japanese
     * @struct col_major
     * @brief 2Dメモリ内データレイアウトが列優先であることを示すメタタグ。
     *
     * \~
     */
    struct col_major
    {
    };

    /**
     * \~english
     * @struct matrix_a
     * @brief Meta-tag indicating data context is input Matrix A.
     *
     * \~chinese
     * @struct matrix_a
     * @brief 元标记，表示数据上下文为输入矩阵A。
     *
     * \~japanese
     * @struct matrix_a
     * @brief データコンテキストが入力行列Aであることを示すメタタグ。
     *
     * \~
     */
    struct matrix_a
    {
    };

    /**
     * \~english
     * @struct matrix_b
     * @brief Meta-tag indicating data context is input Matrix B.
     *
     * \~chinese
     * @struct matrix_b
     * @brief 元标记，表示数据上下文为输入矩阵B。
     *
     * \~japanese
     * @struct matrix_b
     * @brief データコンテキストが入力行列Bであることを示すメタタグ。
     *
     * \~
     */
    struct matrix_b
    {
    };

    /**
     * \~english
     * @struct accumulator
     * @brief Meta-tag indicating data context is Accumulator (also used as Matrix C / D).
     *
     * \~chinese
     * @struct accumulator
     * @brief 元标记，表示数据上下文为累加器（也用作矩阵C/D）。
     *
     * \~japanese
     * @struct accumulator
     * @brief データコンテキストがアキュムレータ（行列C/Dとしても使用）であることを示すメタタグ。
     *
     * \~
     */
    struct accumulator
    {
    };

    /**
     * \~english
     * @struct layout_t
     * @brief Runtime data layout tags
     * @var mem_row_major
     * @var mem_col_major
     *
     * \~chinese
     * @struct layout_t
     * @brief 运行时数据布局标记
     * @var mem_row_major 内存行主序
     * @var mem_col_major 内存列主序
     *
     * \~japanese
     * @struct layout_t
     * @brief ランタイムデータレイアウトタグ
     * @var mem_row_major メモリ行優先
     * @var mem_col_major メモリ列優先
     *
     * \~
     */
    enum layout_t : uint32_t
    {
        mem_row_major,
        mem_col_major
    };

    namespace fragment_scheduler
    {
        /**
         * \~english
         * @typedef default_schedule
         * @brief The default fragment scheduler; each wave operates independently.
         *
         * \~chinese
         * @typedef default_schedule
         * @brief 默认片段调度器；每个波前独立运行。
         *
         * \~japanese
         * @typedef default_schedule
         * @brief デフォルトのフラグメントスケジューラ。各ウェーブは独立して動作します。
         *
         * \~
         */
        using default_schedule = IOScheduler::Default;

        /**
         * \~english
         * @typedef coop_row_major_2d
         * @brief  A cooperative scheduling strategy where each wave in the 2d threadblock
         * will contribute to the fragment operation in row_major grid order.
         * All waves are scheduled in row_major order.
         * E.g. (TBlockX, TBlockY) => 2x2 waves
         * w0 = (0, 0),  w1 = (0, 1),
         * w2 = (1, 0),  w3 = (1, 1)
         * @tparam TBlockX the size of the thread-block in the X dimension
         * @tparam TBlockY the size of the thread-block in the Y dimension
         *
         * \~chinese
         * @typedef coop_row_major_2d
         * @brief 协作调度策略，其中2D线程块中的每个波前将按行主序网格顺序
         * 参与片段操作。所有波前按行主序调度。
         * 例如 (TBlockX, TBlockY) => 2x2个波前
         * w0 = (0, 0),  w1 = (0, 1),
         * w2 = (1, 0),  w3 = (1, 1)
         * @tparam TBlockX X维度上的线程块大小
         * @tparam TBlockY Y维度上的线程块大小
         *
         * \~japanese
         * @typedef coop_row_major_2d
         * @brief 2Dスレッドブロック内の各ウェーブが行優先グリッド順序で
         * フラグメント操作に貢献する協調スケジューリング戦略。
         * すべてのウェーブは行優先順序でスケジュールされます。
         * 例: (TBlockX, TBlockY) => 2x2ウェーブ
         * w0 = (0, 0),  w1 = (0, 1),
         * w2 = (1, 0),  w3 = (1, 1)
         * @tparam TBlockX X次元のスレッドブロックサイズ
         * @tparam TBlockY Y次元のスレッドブロックサイズ
         *
         * \~
         */
        template <uint32_t TBlockX, uint32_t TBlockY>
        using coop_row_major_2d = IOScheduler::RowMajor2d<TBlockX, TBlockY>;

        /**
         * \~english
         * @typedef coop_col_major_2d
         * @brief  A cooperative scheduling strategy where each wave in the 2d threadblock
         * will contribute to the fragment operation in col_major grid order.
         * All waves are scheduled in row_major order.
         * E.g. (TBlockX, TBlockY) => 2x2 waves
         * w0 = (0, 0),  w2 = (0, 1),
         * w1 = (1, 0),  w3 = (1, 1)
         * @tparam TBlockX the size of the thread-block in the X dimension
         * @tparam TBlockY the size of the thread-block in the Y dimension
         *
         * \~chinese
         * @typedef coop_col_major_2d
         * @brief 协作调度策略，其中2D线程块中的每个波前将按列主序网格顺序
         * 参与片段操作。所有波前按行主序调度。
         * 例如 (TBlockX, TBlockY) => 2x2个波前
         * w0 = (0, 0),  w2 = (0, 1),
         * w1 = (1, 0),  w3 = (1, 1)
         * @tparam TBlockX X维度上的线程块大小
         * @tparam TBlockY Y维度上的线程块大小
         *
         * \~japanese
         * @typedef coop_col_major_2d
         * @brief 2Dスレッドブロック内の各ウェーブが列優先グリッド順序で
         * フラグメント操作に貢献する協調スケジューリング戦略。
         * すべてのウェーブは行優先順序でスケジュールされます。
         * 例: (TBlockX, TBlockY) => 2x2ウェーブ
         * w0 = (0, 0),  w2 = (0, 1),
         * w1 = (1, 0),  w3 = (1, 1)
         * @tparam TBlockX X次元のスレッドブロックサイズ
         * @tparam TBlockY Y次元のスレッドブロックサイズ
         *
         * \~
         */
        template <uint32_t TBlockX, uint32_t TBlockY>
        using coop_col_major_2d = IOScheduler::ColMajor2d<TBlockX, TBlockY>;

        /**
         * \~english
         * @typedef coop_row_slice_2d
         * @brief  A cooperative scheduling strategy where each row of waves
         * in the 2d threadblock will contribute to the fragment operation.
         * Waves are partitioned into rows. Only waves in the same row
         * participate together.
         * E.g. (TBlockX, TBlockY) = 2x2 waves
         * RowSlice0: w0 = (0, 0), w1 = (0, 1)
         * RowSlice1: w0 = (1, 0), w1 = (1, 1)
         * @tparam TBlockX the size of the thread-block in the X dimension
         * @tparam TBlockY the size of the thread-block in the Y dimension
         *
         * \~chinese
         * @typedef coop_row_slice_2d
         * @brief 协作调度策略，其中2D线程块中的每一行波前
         * 将参与片段操作。波前被划分为行。只有同一行中的波前
         * 一起参与。
         * 例如 (TBlockX, TBlockY) = 2x2个波前
         * RowSlice0: w0 = (0, 0), w1 = (0, 1)
         * RowSlice1: w0 = (1, 0), w1 = (1, 1)
         * @tparam TBlockX X维度上的线程块大小
         * @tparam TBlockY Y维度上的线程块大小
         *
         * \~japanese
         * @typedef coop_row_slice_2d
         * @brief 2Dスレッドブロック内の各行のウェーブが
         * フラグメント操作に貢献する協調スケジューリング戦略。
         * ウェーブは行に分割されます。同じ行のウェーブのみが
         * 一緒に参加します。
         * 例: (TBlockX, TBlockY) = 2x2ウェーブ
         * RowSlice0: w0 = (0, 0), w1 = (0, 1)
         * RowSlice1: w0 = (1, 0), w1 = (1, 1)
         * @tparam TBlockX X次元のスレッドブロックサイズ
         * @tparam TBlockY Y次元のスレッドブロックサイズ
         *
         * \~
         */
        template <uint32_t TBlockX, uint32_t TBlockY>
        using coop_row_slice_2d = IOScheduler::RowSlice2d<TBlockX, TBlockY>;

        /**
         * \~english
         * @typedef coop_col_slice_2d
         * @brief  A cooperative scheduling strategy where each col of waves
         * in the 2d threadblock will contribute to the fragment operation.
         * Waves are partitioned into cols. Only waves in the same col
         * participate together.
         * E.g. (TBlockX, TBlockY) = 2x2 waves
         * ColSlice0:     ColSlice1:
         * w0 = (0, 0),   w0 = (0, 1),
         * w1 = (1, 0)    w1 = (1, 1)
         * @tparam TBlockX the size of the thread-block in the X dimension
         * @tparam TBlockY the size of the thread-block in the Y dimension
         *
         * \~chinese
         * @typedef coop_col_slice_2d
         * @brief 协作调度策略，其中2D线程块中的每一列波前
         * 将参与片段操作。波前被划分为列。只有同一列中的波前
         * 一起参与。
         * 例如 (TBlockX, TBlockY) = 2x2个波前
         * ColSlice0:     ColSlice1:
         * w0 = (0, 0),   w0 = (0, 1),
         * w1 = (1, 0)    w1 = (1, 1)
         * @tparam TBlockX X维度上的线程块大小
         * @tparam TBlockY Y维度上的线程块大小
         *
         * \~japanese
         * @typedef coop_col_slice_2d
         * @brief 2Dスレッドブロック内の各列のウェーブが
         * フラグメント操作に貢献する協調スケジューリング戦略。
         * ウェーブは列に分割されます。同じ列のウェーブのみが
         * 一緒に参加します。
         * 例: (TBlockX, TBlockY) = 2x2ウェーブ
         * ColSlice0:     ColSlice1:
         * w0 = (0, 0),   w0 = (0, 1),
         * w1 = (1, 0)    w1 = (1, 1)
         * @tparam TBlockX X次元のスレッドブロックサイズ
         * @tparam TBlockY Y次元のスレッドブロックサイズ
         *
         * \~
         */
        template <uint32_t TBlockX, uint32_t TBlockY>
        using coop_col_slice_2d = IOScheduler::ColSlice2d<TBlockX, TBlockY>;

        /**
         * \~english
         * @typedef single
         * @brief  A cooperative scheduling strategy where only one wave in
         * the thread block will participate.
         * @tparam TBlockX the size of the thread-block in the X dimension
         * @tparam TBlockY the size of the thread-block in the Y dimension
         * @tparam WaveIdx the index of the wave which will participate
         *
         * \~chinese
         * @typedef single
         * @brief 协作调度策略，其中线程块中只有一个波前
         * 将参与。
         * @tparam TBlockX X维度上的线程块大小
         * @tparam TBlockY Y维度上的线程块大小
         * @tparam WaveIdx 将参与的波前索引
         *
         * \~japanese
         * @typedef single
         * @brief スレッドブロック内の1つのウェーブのみが
         * 参加する協調スケジューリング戦略。
         * @tparam TBlockX X次元のスレッドブロックサイズ
         * @tparam TBlockY Y次元のスレッドブロックサイズ
         * @tparam WaveIdx 参加するウェーブのインデックス
         *
         * \~
         */
        template <uint32_t TBlockX, uint32_t TBlockY, uint32_t WaveIdx = 0u>
        using single = IOScheduler::Single<TBlockX, TBlockY, WaveIdx>;

    } // namespace fragment_scheduler

    /**
     * \~english
     * @class fragment
     * @brief rocWMMA fragment class. This is the primary object used in block-wise decomposition of the matrix multiply-accumulate (mma)
     * problem space. In general, fragment data is associated with a matrix context (matrix_a, matrix_b or accumulator), a block size (BlockM/N/K),
     * a datatype (e.g. single-precision float, etc.) and an in-memory 2D layout (e.g. row_major or col_major). These fragment properties are used
     * to define how data is handled and stored locally, and to drive API implementations for loading / storing, mma and transforms. Fragment abstractions are
     * designed to promote a simple wavefront programming model, which can accelerate development time. Internal thread-level details are handled by rocWMMA
     * which frees the user to focus on wavefront block-wise decomposition. Written purely in device code, the programmer can use this object in their own
     * device kernels.
     *
     * @tparam MatrixT fragment context
     * @tparam FragM/N/K fragment dimensions
     * @tparam DataT datatype
     * @tparam DataLayoutT in-memory layout as col_major or row_major
     * @tparam Scheduler wave-wise scheduler
     * @note Fragments are stored in packed registers, however vector elements have no guaranteed order or locality.
     *
     * \~chinese
     * @class fragment
     * @brief rocWMMA片段类。这是用于矩阵乘累加（mma）问题空间的块级分解的主要对象。
     * 一般来说，片段数据与矩阵上下文（matrix_a、matrix_b或accumulator）、块大小（BlockM/N/K）、
     * 数据类型（例如单精度浮点数等）和内存中的2D布局（例如row_major或col_major）相关联。这些片段属性用于
     * 定义如何在本地处理和存储数据，并驱动加载/存储、mma和转换的API实现。片段抽象
     * 旨在促进简单的波前编程模型，这可以加快开发时间。内部线程级细节由rocWMMA处理，
     * 这使用户可以专注于波前块级分解。完全用设备代码编写，程序员可以在他们自己的设备内核中使用此对象。
     *
     * @tparam MatrixT 片段上下文
     * @tparam FragM/N/K 片段维度
     * @tparam DataT 数据类型
     * @tparam DataLayoutT 内存布局，col_major或row_major
     * @tparam Scheduler 波前级调度器
     * @note 片段存储在打包寄存器中，但向量元素没有保证的顺序或局部性。
     *
     * \~japanese
     * @class fragment
     * @brief rocWMMAフラグメントクラス。これは、行列乗算累積（mma）問題空間のブロック単位分解で使用される主要なオブジェクトです。
     * 一般に、フラグメントデータは、行列コンテキスト（matrix_a、matrix_b、またはaccumulator）、ブロックサイズ（BlockM/N/K）、
     * データ型（単精度浮動小数点数など）、およびメモリ内2Dレイアウト（row_majorまたはcol_majorなど）に関連付けられています。これらのフラグメントプロパティは、
     * データをローカルで処理および保存する方法を定義し、ロード/ストア、mma、および変換のAPI実装を駆動するために使用されます。フラグメント抽象化は、
     * 開発時間を短縮できるシンプルなウェーブフロントプログラミングモデルを促進するように設計されています。内部スレッドレベルの詳細はrocWMMAによって処理され、
     * ユーザーはウェーブフロントのブロック単位分解に集中できます。純粋にデバイスコードで記述されており、プログラマーは独自のデバイスカーネルでこのオブジェクトを使用できます。
     *
     * @tparam MatrixT フラグメントコンテキスト
     * @tparam FragM/N/K フラグメント次元
     * @tparam DataT データ型
     * @tparam DataLayoutT メモリ内レイアウト（col_majorまたはrow_major）
     * @tparam Scheduler ウェーブ単位スケジューラ
     * @note フラグメントはパックされたレジスタに格納されますが、ベクトル要素には保証された順序や局所性がありません。
     *
     * \~
     */
    template <typename MatrixT,
              uint32_t FragM,
              uint32_t FragN,
              uint32_t FragK,
              typename DataT,
              typename DataLayoutT = void,
              typename Scheduler   = fragment_scheduler::default_schedule>
    class __align__(4) fragment
    {
    public:
        //! Input / output traits specific to AMDGCN architecture
        using IOTraits =
            typename IOConfig<MatrixT, FragM, FragN, FragK, DataT, DataLayoutT, Scheduler>::
                IOTraits;

        struct Traits
        {
        private:
            //! The packed type for element data
            using PackedElementT = typename PackTraits<DataT>::PackedT;

            //! The unpacked type for element data
            using UnpackedElementT = typename PackTraits<DataT>::UnpackedT;

            //! WaveCount sizing factor for cooperative fragments
            static constexpr uint32_t WaveCount = scheduler_traits<Scheduler>::WaveCount;

            //! Assert the fragment occupies at least one packed register
            static_assert(IOTraits::PackedVRegCount >= 1,
                          "Fragments must occupy at least one packed register");

            //! Assert the fragment is equally splittable among the wave count
            static_assert(WaveCount >= 1, "WaveCount must be at least 1 for a valid fragment");
            static_assert(IOTraits::PackedVRegCount >= WaveCount,
                          "Packed register count must be equal to or greater than WaveCount");
            static_assert(IOTraits::PackedVRegCount % WaveCount == 0,
                          "Packed register count must be divisible by WaveCount");

        public:
            constexpr static uint32_t Size = IOTraits::UnpackedSize / WaveCount;

            //! Unpacked data access view
            using AccessT = VecT<UnpackedElementT, Size>;

            //! Packed data storage view
            using StorageT = VecT<PackedElementT, IOTraits::PackedSize / WaveCount>;

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

    /**
     * \~english
     * @brief Fills the entire fragment with the desired value.
     * @param frag Fragment of type MatrixT with its associated block sizes, data type and layout
     * @param value Fill value of type DataT
     * @tparam FragT Opaque fragment type
     * @tparam DataT Datatype
     *
     * \~chinese
     * @brief 用指定值填充整个片段。
     * @param frag MatrixT类型的片段，包含其关联的块大小、数据类型和布局
     * @param value DataT类型的填充值
     * @tparam FragT 不透明片段类型
     * @tparam DataT 数据类型
     *
     * \~japanese
     * @brief フラグメント全体を指定された値で埋める。
     * @param frag MatrixT型のフラグメント（関連するブロックサイズ、データ型、レイアウトを含む）
     * @param value DataT型の埋め込み値
     * @tparam FragT 不透明なフラグメント型
     * @tparam DataT データ型
     *
     * \~
     */
    template <typename FragT, typename DataT>
    ROCWMMA_DEVICE void fill_fragment(FragT& frag, DataT value);

    /**
     * \~english
     * @brief Loads the entire fragment from the data pointer according to its matrix and data layout contexts.
     * Data pointer may point to either local or global memory.
     * @param frag Fragment of type MatrixT with its associated block sizes, data type and layout
     * @param data Data pointer to global or local memory
     * @param ldm Leading dimension size
     * @tparam FragT Opaque fragment type
     * @tparam DataT Datatype
     *
     * \~chinese
     * @brief 根据矩阵和数据布局上下文从数据指针加载整个片段。
     * 数据指针可以指向本地或全局内存。
     * @param frag MatrixT类型的片段，包含其关联的块大小、数据类型和布局
     * @param data 指向全局或本地内存的数据指针
     * @param ldm 主维度大小
     * @tparam FragT 不透明片段类型
     * @tparam DataT 数据类型
     *
     * \~japanese
     * @brief 行列およびデータレイアウトコンテキストに従って、データポインタからフラグメント全体をロードする。
     * データポインタは、ローカルメモリまたはグローバルメモリを指すことができます。
     * @param frag MatrixT型のフラグメント（関連するブロックサイズ、データ型、レイアウトを含む）
     * @param data グローバルまたはローカルメモリへのデータポインタ
     * @param ldm 主次元サイズ
     * @tparam FragT 不透明なフラグメント型
     * @tparam DataT データ型
     *
     * \~
     */
    template <typename FragT, typename DataT>
    ROCWMMA_DEVICE void load_matrix_sync(FragT& frag, const DataT* data, uint32_t ldm);

    /**
     * \~english
     * @brief Loads the entire fragment from the data pointer according to its matrix layout and data layout contexts.
     * Data pointer may point to either local or global memory. This overload provides manual selection of data layout of the incoming memory pointer,
     * which will be transformed to conform to the data layout of the fragment.
     * @param frag Fragment of type MatrixT with its associated block sizes, data type and layout
     * @param data Data pointer to global/local memory
     * @param ldm Leading dimension size
     * @param layout Data layout
     * @tparam FragT Opaque fragment type
     * @tparam DataT Datatype
     *
     * \~chinese
     * @brief 根据矩阵布局和数据布局上下文从数据指针加载整个片段。
     * 数据指针可以指向本地或全局内存。此重载提供对传入内存指针的数据布局的手动选择，
     * 该布局将被转换以符合片段的数据布局。
     * @param frag MatrixT类型的片段，包含其关联的块大小、数据类型和布局
     * @param data 指向全局/本地内存的数据指针
     * @param ldm 主维度大小
     * @param layout 数据布局
     * @tparam FragT 不透明片段类型
     * @tparam DataT 数据类型
     *
     * \~japanese
     * @brief 行列レイアウトおよびデータレイアウトコンテキストに従って、データポインタからフラグメント全体をロードする。
     * データポインタは、ローカルメモリまたはグローバルメモリを指すことができます。このオーバーロードは、受信メモリポインタのデータレイアウトを手動で選択し、
     * フラグメントのデータレイアウトに準拠するように変換します。
     * @param frag MatrixT型のフラグメント（関連するブロックサイズ、データ型、レイアウトを含む）
     * @param data グローバル/ローカルメモリへのデータポインタ
     * @param ldm 主次元サイズ
     * @param layout データレイアウト
     * @tparam FragT 不透明なフラグメント型
     * @tparam DataT データ型
     *
     * \~
     */
    template <typename FragT, typename DataT>
    ROCWMMA_DEVICE void
        load_matrix_sync(FragT& frag, const DataT* data, uint32_t ldm, layout_t layout);

    /**
     * \~english
     * @brief Stores the entire fragment to the data pointer according to its matrix and data layouts.
     * Data pointer may point to either local or global memory.
     * @param data Data pointer to global/local memory
     * @param frag Fragment of type MatrixT with its associated block sizes, data type and layout
     * @param ldm Leading dimension size
     * @tparam DataT Datatype
     * @tparam FragT Opaque fragment type
     *
     * \~chinese
     * @brief 根据矩阵和数据布局将整个片段存储到数据指针。
     * 数据指针可以指向本地或全局内存。
     * @param data 指向全局/本地内存的数据指针
     * @param frag MatrixT类型的片段，包含其关联的块大小、数据类型和布局
     * @param ldm 主维度大小
     * @tparam DataT 数据类型
     * @tparam FragT 不透明片段类型
     *
     * \~japanese
     * @brief 行列およびデータレイアウトに従って、フラグメント全体をデータポインタに格納する。
     * データポインタは、ローカルメモリまたはグローバルメモリを指すことができます。
     * @param data グローバル/ローカルメモリへのデータポインタ
     * @param frag MatrixT型のフラグメント（関連するブロックサイズ、データ型、レイアウトを含む）
     * @param ldm 主次元サイズ
     * @tparam DataT データ型
     * @tparam FragT 不透明なフラグメント型
     *
     * \~
     */
    template <typename FragT, typename DataT>
    ROCWMMA_DEVICE void store_matrix_sync(DataT* data, FragT const& frag, uint32_t ldm);

    /**
     * \~english
     * @brief Stores the entire fragment to the data pointer according to its matrix layout and data layout contexts.
     * Data pointer may point to either local or global memory. This overload provides manual selection of data layout of the outgoing memory pointer,
     * which the data layout of the fragment will be transformed to.
     * @param data Data pointer to global/local memory
     * @param frag Fragment of type MatrixT with its associated block sizes, data type and layout
     * @param ldm Leading dimension size
     * @param layout Data layout
     * @tparam DataT Datatype
     * @tparam FragT Opaque fragment type
     *
     * \~chinese
     * @brief 根据矩阵布局和数据布局上下文将整个片段存储到数据指针。
     * 数据指针可以指向本地或全局内存。此重载提供对传出内存指针的数据布局的手动选择，
     * 片段的数据布局将被转换为该布局。
     * @param data 指向全局/本地内存的数据指针
     * @param frag MatrixT类型的片段，包含其关联的块大小、数据类型和布局
     * @param ldm 主维度大小
     * @param layout 数据布局
     * @tparam DataT 数据类型
     * @tparam FragT 不透明片段类型
     *
     * \~japanese
     * @brief 行列レイアウトおよびデータレイアウトコンテキストに従って、フラグメント全体をデータポインタに格納する。
     * データポインタは、ローカルメモリまたはグローバルメモリを指すことができます。このオーバーロードは、送信メモリポインタのデータレイアウトを手動で選択し、
     * フラグメントのデータレイアウトがそのレイアウトに変換されます。
     * @param data グローバル/ローカルメモリへのデータポインタ
     * @param frag MatrixT型のフラグメント（関連するブロックサイズ、データ型、レイアウトを含む）
     * @param ldm 主次元サイズ
     * @param layout データレイアウト
     * @tparam DataT データ型
     * @tparam FragT 不透明なフラグメント型
     *
     * \~
     */
    template <typename FragT, typename DataT>
    ROCWMMA_DEVICE void
        store_matrix_sync(DataT* data, FragT const& frag, uint32_t ldm, layout_t layout);

    /**
     * \~english
     * @brief Performs the Multiply-Accumulate operation on the fragments A, B, C and D (D = A * B + C)
     * @param d Accumulator output D
     * @param a Input fragment A
     * @param b Input fragment B
     * @param c Input accumulator fragment C
     * @tparam FragA Opaque fragment type for matrix A data
     * @tparam FragB Opaque fragment type for matrix B data
     * @tparam FragAccumIn Opaque fragment type for input accumulation data
     * @tparam FragAccumOut Opaque fragment type for output accumulation data
     * @note Frag c = d is valid
     *
     * \~chinese
     * @brief 对片段A、B、C和D执行乘累加操作（D = A * B + C）
     * @param d 累加器输出D
     * @param a 输入片段A
     * @param b 输入片段B
     * @param c 输入累加器片段C
     * @tparam FragA 矩阵A数据的不透明片段类型
     * @tparam FragB 矩阵B数据的不透明片段类型
     * @tparam FragAccumIn 输入累加数据的不透明片段类型
     * @tparam FragAccumOut 输出累加数据的不透明片段类型
     * @note 片段c = d是有效的
     *
     * \~japanese
     * @brief フラグメントA、B、C、Dに対して乗算累積演算を実行する（D = A * B + C）
     * @param d アキュムレータ出力D
     * @param a 入力フラグメントA
     * @param b 入力フラグメントB
     * @param c 入力アキュムレータフラグメントC
     * @tparam FragA 行列Aデータの不透明なフラグメント型
     * @tparam FragB 行列Bデータの不透明なフラグメント型
     * @tparam FragAccumIn 入力累積データの不透明なフラグメント型
     * @tparam FragAccumOut 出力累積データの不透明なフラグメント型
     * @note フラグメントc = dは有効です
     *
     * \~
     */
    template <typename FragA, typename FragB, typename FragAccumIn, typename FragAccumOut>
    ROCWMMA_DEVICE void mma_sync(FragAccumOut& d, FragA const& a, FragB const& b, FragAccumIn& c);

    /**
     * \~english
     * @brief Synchronization point for all wavefronts in a workgroup.
     * Guarantees pending reads / writes to LDS are flushed.
     *
     * \~chinese
     * @brief 工作组中所有波前的同步点。
     * 保证待处理的LDS读/写被刷新。
     *
     * \~japanese
     * @brief ワークグループ内のすべてのウェーブフロントの同期ポイント。
     * 保留中のLDSへの読み取り/書き込みがフラッシュされることを保証します。
     *
     * \~
     */
    ROCWMMA_DEVICE ROCWMMA_INLINE void synchronize_workgroup();

    /** @}*/
} // namespace rocwmma

#include "rocwmma_impl.hpp"

#endif // ROCWMMA_API_HPP
