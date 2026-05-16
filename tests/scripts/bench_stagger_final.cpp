// bench_stagger_final.cpp — Wave staggering parameter sweep
//
// Task B: Test atomic staggering with different wave counts,
// launch multipliers, tile counts, and work distribution patterns.
//
// Build:
//   /opt/llvm-amd/bin/clang++ -x hip --offload-arch=gfx1200 \
//     -I/home/yanli/work/ROCm/rocWMMA/library/include \
//     -I/opt/rocm/include -DROCWMMA_WAVE32_MODE=1 -O3 \
//     -L/opt/rocm/lib -lamdhip64 -o bench_stagger_final bench_stagger_final.cpp

#include <rocwmma/rocwmma_16chain.hpp>
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
static constexpr double O = 32768.0;
static constexpr int LO = 160, TT = 1024;

// ======================================================================
// K: Lightweight atomic staggered kernel
//    workers = waves that do work, launch_mult = oversubscription factor
// ======================================================================
template<int CH, int TI>
__global__ __launch_bounds__(32,2)
void k(int32_t*C,const int32_t*A,const int32_t*B,int L,int* counter,int MAX_WORK){
    int claimed=atomicAdd(counter,1);
    if(claimed>=MAX_WORK)return;
    int w=claimed;
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
// Benchmark
// ======================================================================
double bench(int workers,int tiles,int L,int it,int launch_mult,int CH){
    int launch=workers*launch_mult, TI=tiles, bsz=workers*tiles*4;
    int32_t*dC,*dA,*dB,*cnt;
    hipMalloc(&dC,(size_t)workers*CH*tiles*8*4);
    hipMalloc(&dA,(size_t)workers*2*4);
    hipMalloc(&dB,(size_t)bsz*4);
    hipMalloc(&cnt,4);
    std::vector<int32_t>hA(workers*2,0x32103210),hB(bsz,0x76547654);
    hipMemcpy(dA,hA.data(),(size_t)workers*8,hipMemcpyHostToDevice);
    hipMemcpy(dB,hB.data(),(size_t)bsz*4,hipMemcpyHostToDevice);
    // warmup
    for(int i=0;i<3;++i){
        hipMemset(cnt,0,4);
        if(CH==16&&TI==1)k<16,1><<<launch,32>>>(dC,dA,dB,L,cnt,workers);
        else if(CH==16&&TI==2)k<16,2><<<launch,32>>>(dC,dA,dB,L,cnt,workers);
        else if(CH==16&&TI==4)k<16,4><<<launch,32>>>(dC,dA,dB,L,cnt,workers);
        else if(CH==14&&TI==1)k<14,1><<<launch,32>>>(dC,dA,dB,L,cnt,workers);
        else if(CH==14&&TI==2)k<14,2><<<launch,32>>>(dC,dA,dB,L,cnt,workers);
        else k<14,4><<<launch,32>>>(dC,dA,dB,L,cnt,workers);
    }
    hipDeviceSynchronize();

    hipEvent_t s,e;hipEventCreate(&s);hipEventCreate(&e);hipEventRecord(s,0);
    for(int i=0;i<it;++i){
        hipMemset(cnt,0,4);
        if(CH==16&&TI==1)k<16,1><<<launch,32>>>(dC,dA,dB,L,cnt,workers);
        else if(CH==16&&TI==2)k<16,2><<<launch,32>>>(dC,dA,dB,L,cnt,workers);
        else if(CH==16&&TI==4)k<16,4><<<launch,32>>>(dC,dA,dB,L,cnt,workers);
        else if(CH==14&&TI==1)k<14,1><<<launch,32>>>(dC,dA,dB,L,cnt,workers);
        else if(CH==14&&TI==2)k<14,2><<<launch,32>>>(dC,dA,dB,L,cnt,workers);
        else k<14,4><<<launch,32>>>(dC,dA,dB,L,cnt,workers);
    }
    hipDeviceSynchronize();
    hipEventRecord(e,0);hipEventSynchronize(e);
    float ms;hipEventElapsedTime(&ms,s,e);
    hipEventDestroy(s);hipEventDestroy(e);
    hipFree(dC);hipFree(dA);hipFree(dB);hipFree(cnt);
    return O*(double)workers*CH*tiles*L/(ms/(double)it*1e-3)/1e12;
}

int main(){
    hipDeviceProp_t p;hipGetDeviceProperties(&p,0);
    int nCu=p.multiProcessorCount*2;
    printf("GPU: %s (%d CUs, %d SIMDs)\n\n",p.name,nCu,nCu*2);
    int L=LO;

    struct R{char n[64];double t;int w;int tiles;int mult;int ch;};
    std::vector<R> rs;

    // ====================================================================
    // SWEEP 1: Launch multiplier (oversubscription)
    //   1024 working waves, 1 tile, 16ch
    //   mult: 1x, 2x, 4x, 8x, 16x
    // ====================================================================
    printf("=== SWEEP 1: Launch multiplier (1024w, 1t, 16ch) ===\n");
    for(int mult:{1,2,4,8,16}){
        double tp=bench(1024,1,L,10,mult,16);
        printf("  mult=%2dx: %8.0f TOPs  (%.1f%% theory)\n",mult,tp,tp/5830*100);
        R r;snprintf(r.n,sizeof(r.n),"MULT_%dx",mult);
        r.t=tp;r.w=1024;r.tiles=1;r.mult=mult;r.ch=16;rs.push_back(r);
    }

    // ====================================================================
    // SWEEP 2: Wave count sweep (fixed total ops = 1024w worth)
    //   Different wave counts with proportional tiles to keep total ops constant
    // ====================================================================
    printf("\n=== SWEEP 2: Wave count vs tiles (constant total ops) ===\n");
    for(int ww:{64,128,256,512,1024}){
        int tiles=TT/ww, it=(tiles>=8)?8:(tiles>=4)?10:15;
        for(int mult:{1,2}){
            double tp=bench(ww,tiles,L,it,mult,16);
            printf("  %4dw x %dt mult=%dx: %8.0f TOPs  (%.1f%% theory, total waves=%d)\n",
                   ww,tiles,mult,tp,tp/5830*100,ww*mult);
            R r;snprintf(r.n,sizeof(r.n),"WS_%dw_%dt_M%d",ww,tiles,mult);
            r.t=tp;r.w=ww;r.tiles=tiles;r.mult=mult;r.ch=16;rs.push_back(r);
        }
    }

    // ====================================================================
    // SWEEP 3: 14ch vs 16ch comparison at best configs
    // ====================================================================
    printf("\n=== SWEEP 3: 14ch vs 16ch (1024w, 1t) ===\n");
    for(int mult:{1,2,4}){
        double tp16=bench(1024,1,L,10,mult,16);
        double tp14=bench(1024,1,L,10,mult,14);
        printf("  mult=%dx: 16ch=%.0f  14ch=%.0f  diff=%+.0f (%.1f%%)\n",
               mult,tp16,tp14,tp14-tp16,(tp14/tp16-1)*100);
        R r;snprintf(r.n,sizeof(r.n),"CH_14ch_M%d",mult);
        r.t=tp14;r.w=1024;r.tiles=1;r.mult=mult;r.ch=14;rs.push_back(r);
    }

    // ====================================================================
    // SWEEP 4: Tile count vs wave count trade-off (fixed total work)
    // ====================================================================
    printf("\n=== SWEEP 4: Fixed total work (1024 waves or equivalent) ===\n");
    for(int ww:{1024,512,256}){
        int tiles=TT/ww;
        for(int mult:{1,2}){
            double tp=bench(ww,tiles,L,(tiles>=4)?10:15,mult,16);
            printf("  %4dw x %dt mult=%d: %8.0f TOPs  (waves/tile*cmbn=%d)\n",
                   ww,tiles,mult,tp,ww*tiles);
            R r;snprintf(r.n,sizeof(r.n),"WT_%dw_%dt_M%d",ww,tiles,mult);
            r.t=tp;r.w=ww;r.tiles=tiles;r.mult=mult;r.ch=16;rs.push_back(r);
        }
    }

    // ====================================================================
    // SORT AND DISPLAY BEST
    // ====================================================================
    std::sort(rs.begin(),rs.end(),[](R&a,R&b){return a.t>b.t;});
    printf("\n=== TOP 10 CONFIGS ===\n");
    for(int i=0;i<10&&i<(int)rs.size();++i){
        auto&r=rs[i];
        printf("  %2d. %-20s %8.0f TOPs  (%dw_%dt_%dch_%dxmult)\n",
               i+1,r.n,r.t,r.w,r.tiles,r.ch,r.mult);
    }

    printf("\n=== OPTIMAL CONFIG ===\n");
    auto&best=rs[0];
    printf("  %s: %.0f TOPs = %.1f%% of theoretical 5830 TOPs\n",
           best.n,best.t,best.t/5830*100);
    printf("  IPC: %.3f SWMMAC/cycle/SIMD\n",best.t/5830);
    printf("  Config: %d workers, %d tiles, %dch, %dx launch\n",
           best.w,best.tiles,best.ch,best.mult);
    return 0;
}
