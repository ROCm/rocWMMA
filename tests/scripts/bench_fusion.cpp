// bench_fusion.cpp — Pipeline fusion: SWMMAC + bias + scale + act
//
// Measures wall-clock speedup from fusing post-processing into SWMMAC.
// Hypothesis: saving intermediate store→load roundtrip + kernel launch.
//
// Build:
//   /opt/llvm-amd/bin/clang++ -x hip --offload-arch=gfx1200 \
//     -I/home/yanli/work/ROCm/rocWMMA/library/include \
//     -I/opt/rocm/include -DROCWMMA_WAVE32_MODE=1 -O3 \
//     -L/opt/rocm/lib -lamdhip64 -o bench_fusion bench_fusion.cpp

#include <rocwmma/rocwmma_16chain.hpp>
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
static constexpr double O=32768.0;
static constexpr int LO=160, TT=1024;

// ======================================================================
// K_SWMMAC: atomic staggered SWMMAC → store int32 accumulators
// ======================================================================
template<int CH, int TI>
__global__ __launch_bounds__(32,2)
void k_swmmac(int32_t*C,const int32_t*A,const int32_t*B,int L,int* counter){
    int w=atomicAdd(counter,1);if(w>=TT)return;
    int32_t bt[TI][4];
    for(int t=0;t<TI;++t)for(int j=0;j<4;++j)bt[t][j]=B[(w*TI+t)*4+j];
    alignas(32)int32_t ac[CH][8]={};
    auto&ra=*reinterpret_cast<const rocwmma::SwmmacARegsT*>(A+w*2);
    for(int t=0;t<TI;++t){
        auto&rb=*reinterpret_cast<const rocwmma::SwmmacBRegsT*>(bt[t]);
        for(int i=0;i<L;++i){
            #pragma unroll
            for(int cc=0;cc<CH;++cc){
                auto&rc=*reinterpret_cast<rocwmma::SwmmacAccumT*>(ac[cc]);
                rc=rocwmma::SwmmacI4::exec(ra,rb,rc,0);
            }
        }
    }
    for(int t=0;t<TI;++t)for(int cc=0;cc<CH;++cc)
        for(int j=0;j<8;++j)C[((w*TI+t)*CH+cc)*8+j]=ac[cc][j];
}

// ======================================================================
// K_POST: bias add + scale + ReLU from global memory accumulators
//         (simulates a separate post-processing kernel)
// ======================================================================
__global__ __launch_bounds__(32,2)
void k_post(int32_t*C,const int32_t*bias,float scale,int total_elements){
    int tid=blockIdx.x*32+threadIdx.x;
    int stride=gridDim.x*32;
    for(int i=tid;i<total_elements;i+=stride){
        int32_t val=C[i];
        val += bias[i%16];           // bias per output channel (16 outputs)
        float fval=(float)val*scale; // dequant
        if(fval<0)fval=0;            // ReLU
        C[i]=(int32_t)fval;          // store back (quantized)
    }
}

// ======================================================================
// K_FUSED: SWMMAC + bias + scale + ReLU in one kernel
//          Post-processing done on accumulators WHILE they're in VGPRs
// ======================================================================
template<int CH, int TI>
__global__ __launch_bounds__(32,2)
void k_fused(int32_t*C,const int32_t*A,const int32_t*B,
             const int32_t*bias,float scale,int L,int* counter){
    int w=atomicAdd(counter,1);if(w>=TT)return;
    int32_t bt[TI][4];
    for(int t=0;t<TI;++t)for(int j=0;j<4;++j)bt[t][j]=B[(w*TI+t)*4+j];
    alignas(32)int32_t ac[CH][8]={};
    auto&ra=*reinterpret_cast<const rocwmma::SwmmacARegsT*>(A+w*2);

    // === SWMMAC LOOP (same as k_swmmac) ===
    for(int t=0;t<TI;++t){
        auto&rb=*reinterpret_cast<const rocwmma::SwmmacBRegsT*>(bt[t]);
        for(int i=0;i<L;++i){
            #pragma unroll
            for(int cc=0;cc<CH;++cc){
                auto&rc=*reinterpret_cast<rocwmma::SwmmacAccumT*>(ac[cc]);
                rc=rocwmma::SwmmacI4::exec(ra,rb,rc,0);
            }
        }
    }

    // === FUSED EPILOGUE: bias + scale + ReLU on hot accumulators ===
    for(int t=0;t<TI;++t){
        for(int cc=0;cc<CH;++cc){
            int32_t b=bias[((w*TI+t)*CH+cc)%16]; // 16 bias values, wrap
            for(int j=0;j<8;++j){
                int32_t val=ac[cc][j]+b;
                float fval=(float)val*scale;
                if(fval<0)fval=0;
                C[((w*TI+t)*CH+cc)*8+j]=(int32_t)fval;
            }
        }
    }
}

// ======================================================================
// Benchmark: wall-clock time for complete pipeline
// ======================================================================
struct Result { const char* name; double ms_pipeline; };

Result bench_separate(int workers,int L,int it,int CH,int TI){
    size_t Csz=(size_t)workers*CH*TI*8*4, Asz=(size_t)workers*2*4;
    size_t Bsz=(size_t)workers*TI*4*4;
    int32_t *dC,*dA,*dB,*dbias,*cnt;
    float *dscale;
    hipMalloc(&dC,Csz);hipMalloc(&dA,Asz);hipMalloc(&dB,Bsz);
    hipMalloc(&dbias,16*4);hipMalloc(&dscale,4);
    hipMalloc(&cnt,4);
    std::vector<int32_t>hA(workers*2,0x32103210),hB(workers*TI*4,0x76547654);
    int32_t hbias[16];for(int i=0;i<16;++i)hbias[i]=i*10;
    float hscale=0.0078125f; // 1/128
    hipMemcpy(dA,hA.data(),Asz,hipMemcpyHostToDevice);
    hipMemcpy(dB,hB.data(),Bsz,hipMemcpyHostToDevice);
    hipMemcpy(dbias,hbias,64,hipMemcpyHostToDevice);
    hipMemcpy(dscale,&hscale,4,hipMemcpyHostToDevice);

    // Warmup
    for(int i=0;i<3;++i){
        hipMemset(cnt,0,4);
        if(CH==16&&TI==1)k_swmmac<16,1><<<workers,32>>>(dC,dA,dB,L,cnt);
        else if(CH==16&&TI==2)k_swmmac<16,2><<<workers,32>>>(dC,dA,dB,L,cnt);
        int elems=workers*CH*TI*8;
        int blocks=(elems+31)/32;
        k_post<<<blocks,32>>>(dC,dbias,hscale,elems);
    }
    hipDeviceSynchronize();

    // Timed: complete pipeline = SWMMAC + post
    hipEvent_t s,e;hipEventCreate(&s);hipEventCreate(&e);
    hipEventRecord(s,0);
    for(int i=0;i<it;++i){
        hipMemset(cnt,0,4);
        if(CH==16&&TI==1)k_swmmac<16,1><<<workers,32>>>(dC,dA,dB,L,cnt);
        else if(CH==16&&TI==2)k_swmmac<16,2><<<workers,32>>>(dC,dA,dB,L,cnt);
        int elems=workers*CH*TI*8;
        int blocks=(elems+31)/32;
        k_post<<<blocks,32>>>(dC,dbias,hscale,elems);
    }
    hipDeviceSynchronize();
    hipEventRecord(e,0);hipEventSynchronize(e);
    float ms;hipEventElapsedTime(&ms,s,e);
    hipEventDestroy(s);hipEventDestroy(e);

    hipFree(dC);hipFree(dA);hipFree(dB);hipFree(dbias);hipFree(dscale);hipFree(cnt);
    return {"separate",ms/(double)it};
}

Result bench_fused(int workers,int L,int it,int CH,int TI){
    size_t Csz=(size_t)workers*CH*TI*8*4, Asz=(size_t)workers*2*4;
    size_t Bsz=(size_t)workers*TI*4*4;
    int32_t *dC,*dA,*dB,*dbias,*cnt;
    float *dscale;
    hipMalloc(&dC,Csz);hipMalloc(&dA,Asz);hipMalloc(&dB,Bsz);
    hipMalloc(&dbias,16*4);hipMalloc(&dscale,4);
    hipMalloc(&cnt,4);
    std::vector<int32_t>hA(workers*2,0x32103210),hB(workers*TI*4,0x76547654);
    int32_t hbias[16];for(int i=0;i<16;++i)hbias[i]=i*10;
    float hscale=0.0078125f;
    hipMemcpy(dA,hA.data(),Asz,hipMemcpyHostToDevice);
    hipMemcpy(dB,hB.data(),Bsz,hipMemcpyHostToDevice);
    hipMemcpy(dbias,hbias,64,hipMemcpyHostToDevice);
    hipMemcpy(dscale,&hscale,4,hipMemcpyHostToDevice);

    // Warmup
    for(int i=0;i<3;++i){
        hipMemset(cnt,0,4);
        if(CH==16&&TI==1)k_fused<16,1><<<workers,32>>>(dC,dA,dB,dbias,hscale,L,cnt);
        else if(CH==16&&TI==2)k_fused<16,2><<<workers,32>>>(dC,dA,dB,dbias,hscale,L,cnt);
    }
    hipDeviceSynchronize();

    // Timed
    hipEvent_t s,e;hipEventCreate(&s);hipEventCreate(&e);
    hipEventRecord(s,0);
    for(int i=0;i<it;++i){
        hipMemset(cnt,0,4);
        if(CH==16&&TI==1)k_fused<16,1><<<workers,32>>>(dC,dA,dB,dbias,hscale,L,cnt);
        else if(CH==16&&TI==2)k_fused<16,2><<<workers,32>>>(dC,dA,dB,dbias,hscale,L,cnt);
    }
    hipDeviceSynchronize();
    hipEventRecord(e,0);hipEventSynchronize(e);
    float ms;hipEventElapsedTime(&ms,s,e);
    hipEventDestroy(s);hipEventDestroy(e);

    hipFree(dC);hipFree(dA);hipFree(dB);hipFree(dbias);hipFree(dscale);hipFree(cnt);
    return {"fused  ",ms/(double)it};
}

int main(){
    hipDeviceProp_t p;hipGetDeviceProperties(&p,0);
    printf("GPU: %s (%d CUs)\n",p.name,p.multiProcessorCount*2);
    printf("═══ Pipeline Fusion: SWMMAC + bias + scale + ReLU ═══\n\n");

    int L=LO, it=15;

    for(int workers:{1024,512}){
        int TI=(workers==1024)?1:2;
        printf("--- %d workers, %d tile(s) ---\n",workers,TI);

        // Warm run (stabilize GPU clock)
        bench_separate(workers,L,5,16,TI);

        Result sep = bench_separate(workers,L,it,16,TI);
        Result fus = bench_fused(workers,L,it,16,TI);

        double speedup = (sep.ms_pipeline - fus.ms_pipeline) / sep.ms_pipeline * 100;
        printf("  %s: %8.3f ms/iter  (SWMMAC + post)\n",sep.name,sep.ms_pipeline);
        printf("  %s: %8.3f ms/iter  (fused)\n",fus.name,fus.ms_pipeline);
        printf("  Speedup: %+.1f%%  (saved: %.3f ms)\n\n",speedup,sep.ms_pipeline-fus.ms_pipeline);
    }

    // Also measure: just SWMMAC (no post at all) for reference
    printf("--- SWMMAC-only reference (no post) ---\n");
    Result ref = bench_separate(1024,L,it,16,1);
    // Remove post time: we can't easily, just estimate from the pipeline measurement
    printf("  SWMMAC + post: %.3f ms (includes post overhead)\n", ref.ms_pipeline);

    printf("\n=== Analysis ===\n");
    printf("Fusion eliminates:\n");
    printf("  1. Post-processing kernel launch (~2-4 us)\n");
    printf("  2. Global memory load of accumulators (~524 KB read)\n");
    printf("  3. Additional wave scheduling overhead\n");
    printf("Fusion keeps accumulators in VGPR → bias/scale/act on VALU.\n");
    return 0;
}
