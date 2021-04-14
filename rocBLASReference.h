#ifndef WMMA_ROCBLAS_REFERENCE_H
#define WMMA_ROCBLAS_REFERENCE_H

#define ROCM_USE_FLOAT16

#define CHECK_HIP_ERROR(expr) assert((expr) == hipSuccess);
#define CHECK_ROCBLAS_ERROR(expr) assert((expr) == rocblas_status_success);

#include "rocblas-functions.h"
#include "rocblas.h"

#include "Types.h"

constexpr const char* rocblas_datatype2string(rocblas_datatype type)
{
    switch(type)
    {
    case rocblas_datatype_f16_r:
        return "f16_r";
    case rocblas_datatype_f32_r:
        return "f32_r";
    case rocblas_datatype_f64_r:
        return "f64_r";
    case rocblas_datatype_f16_c:
        return "f16_c";
    case rocblas_datatype_f32_c:
        return "f32_c";
    case rocblas_datatype_f64_c:
        return "f64_c";
    case rocblas_datatype_i8_r:
        return "i8_r";
    case rocblas_datatype_u8_r:
        return "u8_r";
    case rocblas_datatype_i32_r:
        return "i32_r";
    case rocblas_datatype_u32_r:
        return "u32_r";
    case rocblas_datatype_i8_c:
        return "i8_c";
    case rocblas_datatype_u8_c:
        return "u8_c";
    case rocblas_datatype_i32_c:
        return "i32_c";
    case rocblas_datatype_u32_c:
        return "u32_c";
    case rocblas_datatype_bf16_r:
        return "bf16_r";
    case rocblas_datatype_bf16_c:
        return "bf16_c";
    }
    return "invalid";
}

template <typename DataT>
struct rocblas_types;

template <>
struct rocblas_types<float16_t>
{
    using DataType = rocblas_half;
    constexpr static inline rocblas_datatype type()
    {
        return rocblas_datatype_f16_r;
    }
};

template <>
struct rocblas_types<__half> : public rocblas_types<float16_t>
{
};

template <>
struct rocblas_types<float32_t>
{
    using DataType = float32_t;
    constexpr static inline rocblas_datatype type()
    {
        return rocblas_datatype_f32_r;
    }
};

template <typename DataLayoutT>
struct rocblas_layout;

template <>
struct rocblas_layout<row_major>
{
    using Layout = row_major;
    constexpr static inline rocblas_operation operation()
    {
        return rocblas_operation_transpose;
    }
};

template <>
struct rocblas_layout<col_major>
{
    using Layout = col_major;
    constexpr static inline rocblas_operation operation()
    {
        return rocblas_operation_none;
    }
};

/*
* Rocblas notes:
* Layouts C and D are always assumed as col_major
* Non-transpose (N) is col-major
* Transpose (T) is row-major
*/
template <typename InputT, typename OutputT, typename ComputeT, typename LayoutA, typename LayoutB>
void gemm_rocBLAS(uint32_t       m,
                  uint32_t       n,
                  uint32_t       k,
                  InputT const*  ha,
                  InputT const*  hb,
                  OutputT const* hc,
                  OutputT*       hd,
                  ComputeT       alpha,
                  ComputeT       beta)
{
    rocblas_datatype a_type       = rocblas_types<InputT>::type();
    rocblas_datatype b_type       = rocblas_types<InputT>::type();
    rocblas_datatype c_type       = rocblas_types<OutputT>::type();
    rocblas_datatype d_type       = rocblas_types<OutputT>::type();
    rocblas_datatype compute_type = rocblas_types<ComputeT>::type();

    using a_t = typename rocblas_types<InputT>::DataType;
    using b_t = typename rocblas_types<InputT>::DataType;
    using c_t = typename rocblas_types<OutputT>::DataType;
    using d_t = typename rocblas_types<OutputT>::DataType;

    size_t size_a = m * k;
    size_t size_b = k * n;
    size_t size_c = m * n;
    size_t size_d = m * n;

    rocblas_operation opA = rocblas_layout<LayoutA>::operation();
    rocblas_operation opB = rocblas_layout<LayoutB>::operation();

    rocblas_int lda = (opA == rocblas_operation_none ? m : k);
    rocblas_int ldb = (opB == rocblas_operation_none ? k : n);
    rocblas_int ldc = m;
    rocblas_int ldd = m;

    // allocate memory on device
    a_t* da;
    b_t* db;
    c_t* dc;
    d_t* dd;

    CHECK_HIP_ERROR(hipMalloc(&da, size_a * sizeof(a_t)));
    CHECK_HIP_ERROR(hipMalloc(&db, size_b * sizeof(b_t)));
    CHECK_HIP_ERROR(hipMalloc(&dc, size_c * sizeof(c_t)));
    CHECK_HIP_ERROR(hipMalloc(&dd, size_d * sizeof(d_t)));

    // copy matrices from host to device
    CHECK_HIP_ERROR(hipMemcpy(da, &ha[0], sizeof(a_t) * size_a, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(db, &hb[0], sizeof(b_t) * size_b, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dc, &hc[0], sizeof(c_t) * size_c, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dd, &hd[0], sizeof(d_t) * size_d, hipMemcpyHostToDevice));

    rocblas_handle handle;
    CHECK_ROCBLAS_ERROR(rocblas_create_handle(&handle));

    auto     algo           = rocblas_gemm_algo_standard;
    int32_t  solution_index = 0;
    uint32_t flags          = 0;

    CHECK_ROCBLAS_ERROR(rocblas_gemm_ex(handle,
                                        opA,
                                        opB,
                                        m,
                                        n,
                                        k,
                                        &alpha,
                                        da,
                                        a_type,
                                        lda,
                                        db,
                                        b_type,
                                        ldb,
                                        &beta,
                                        dc,
                                        c_type,
                                        ldc,
                                        dd,
                                        d_type,
                                        ldd,
                                        compute_type,
                                        algo,
                                        solution_index,
                                        flags));

    // copy output from device to CPU
    CHECK_HIP_ERROR(hipMemcpy(&hd[0], dd, sizeof(d_t) * size_d, hipMemcpyDeviceToHost));

    // std::cout << "a_type, b_type, c_type, d_type, m, n, k, alpha, lda, "
    //              "ldb, beta, ldc, ldd = "
    //           << rocblas_datatype2string(a_type) << ", " << rocblas_datatype2string(b_type) << ", "
    //           << rocblas_datatype2string(c_type) << ", " << rocblas_datatype2string(d_type) << ", "
    //           << m << ", " << n << ", " << k << ", " << alpha << ", " << lda << ", "
    //           << ldb << ", " << beta << ", " << ldc << ", " << ldd << ", " << std::endl;

    CHECK_HIP_ERROR(hipFree(da));
    CHECK_HIP_ERROR(hipFree(db));
    CHECK_HIP_ERROR(hipFree(dc));
    CHECK_HIP_ERROR(hipFree(dd));
    CHECK_ROCBLAS_ERROR(rocblas_destroy_handle(handle));
}

#endif // WMMA_ROCBLAS_REFERENCE_H
