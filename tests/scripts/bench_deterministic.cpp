// bench_deterministic.cpp — Deterministic wave staggering vs random atomic
//
// Hypothesis: atomicAdd creates variable ~30-cycle wave spacing.
// Deterministic phase offset (NOP delay per wave) should tighten stagger.
// Target: stabilize near IPC=0.700 (the proven hardware peak) vs 0.621 mean.
//
// Build:
//   /opt/llvm-amd/bin/clang++ -x hip --offload-arch=gfx1200 \
//     -I/opt/rocm/include -DROCWMMA_WAVE32_MODE=1 -O3 \
//     -L/opt/rocm/lib -lamdhip64 -o bench_deterministic bench_deterministic.cpp

#include <rocwmma/rocwmma_16chain.hpp>
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
static constexpr double O=32768.0;
static constexpr int LO=160, TT=1024;

// ======================================================================
// K_ATOMIC: baseline — atomicAdd work-claim (our proven K6, ~3621 TOPs mean)
// ======================================================================
__global__ __launch_bounds__(32,2)
void k_atomic(int32_t*C,const int32_t*A,const int32_t*B,int L,int* counter){
    int w=atomicAdd(counter,1);if(w>=TT)return;
    int32_t bt[1][4];
    for(int t=0;t<1;++t)for(int j=0;j<4;++j)bt[t][j]=B[(w+t)*4+j];
    alignas(32)int32_t ac[16][8]={};
    auto&ra=*reinterpret_cast<const rocwmma::SwmmacARegsT*>(A+w*2);
    for(int t=0;t<1;++t){
        auto&rb=*reinterpret_cast<const rocwmma::SwmmacBRegsT*>(bt[t]);
        for(int i=0;i<L;++i){
            #pragma unroll
            for(int cc=0;cc<16;++cc){
                auto&rc=*reinterpret_cast<rocwmma::SwmmacAccumT*>(ac[cc]);
                rc=rocwmma::SwmmacI4::exec(ra,rb,rc,0);
            }
        }
    }
    for(int t=0;t<1;++t)for(int cc=0;cc<16;++cc)
        for(int j=0;j<8;++j)C[((w+t)*16+cc)*8+j]=ac[cc][j];
}

// ======================================================================
// K_PHASE_N: deterministic phase offset via NOP loop
//   Each wave waits (blockIdx.x % 16) * PHASE cycles before starting.
//   Creates 16 phase groups, ideally 1 group per wave slot in each SIMD.
// ======================================================================
template<int PHASE>
__global__ __launch_bounds__(32,2)
void k_phase(int32_t*C,const int32_t*A,const int32_t*B,int L){
    int w=blockIdx.x;if(w>=TT)return;

    // Deterministic phase offset: (w % 16) * PHASE cycles
    int phase = (w & 15) * PHASE;  // w % 16 == w & 15
    #pragma unroll
    for(int p=0;p<phase;++p){__asm__ __volatile__("s_nop 0");}

    int32_t bt[1][4];
    for(int t=0;t<1;++t)for(int j=0;j<4;++j)bt[t][j]=B[(w+t)*4+j];
    alignas(32)int32_t ac[16][8]={};
    auto&ra=*reinterpret_cast<const rocwmma::SwmmacARegsT*>(A+w*2);
    for(int t=0;t<1;++t){
        auto&rb=*reinterpret_cast<const rocwmma::SwmmacBRegsT*>(bt[t]);
        for(int i=0;i<L;++i){
            #pragma unroll
            for(int cc=0;cc<16;++cc){
                auto&rc=*reinterpret_cast<rocwmma::SwmmacAccumT*>(ac[cc]);
                rc=rocwmma::SwmmacI4::exec(ra,rb,rc,0);
            }
        }
    }
    for(int t=0;t<1;++t)for(int cc=0;cc<16;++cc)
        for(int j=0;j<8;++j)C[((w+t)*16+cc)*8+j]=ac[cc][j];
}

// ======================================================================
// K_PHASE_ATOMIC: hybrid — atomic claim for wave count control + phase offset
//   Uses atomic to limit total working waves, then applies phase delay.
// ======================================================================
template<int PHASE>
__global__ __launch_bounds__(32,2)
void k_phase_atomic(int32_t*C,const int32_t*A,const int32_t*B,int L,int* counter){
    int w=atomicAdd(counter,1);if(w>=TT)return;

    // Phase: use the claimed order as stagger key
    int phase = (w & 15) * PHASE;
    #pragma unroll
    for(int p=0;p<phase;++p){__asm__ __volatile__("s_nop 0");}

    int32_t bt[1][4];
    for(int t=0;t<1;++t)for(int j=0;j<4;++j)bt[t][j]=B[(w+t)*4+j];
    alignas(32)int32_t ac[16][8]={};
    auto&ra=*reinterpret_cast<const rocwmma::SwmmacARegsT*>(A+w*2);
    for(int t=0;t<1;++t){
        auto&rb=*reinterpret_cast<const rocwmma::SwmmacBRegsT*>(bt[t]);
        for(int i=0;i<L;++i){
            #pragma unroll
            for(int cc=0;cc<16;++cc){
                auto&rc=*reinterpret_cast<rocwmma::SwmmacAccumT*>(ac[cc]);
                rc=rocwmma::SwmmacI4::exec(ra,rb,rc,0);
            }
        }
    }
    for(int t=0;t<1;++t)for(int cc=0;cc<16;++cc)
        for(int j=0;j<8;++j)C[((w+t)*16+cc)*8+j]=ac[cc][j];
}

// ======================================================================
// Benchmark: sample a kernel multiple times, return stats
// ======================================================================
struct Stats { double min,max,mean,std; int n; };

Stats sample_atomic(int32_t* dC,int32_t* dA,int32_t* dB,int32_t* cnt,int it,int ns){
    std::vector<double> vals;
    for(int s=0;s<ns;++s){
        hipMemset(cnt,0,4); k_atomic<<<1024,32>>>(dC,dA,dB,LO,cnt); hipDeviceSynchronize();
        hipEvent_t e1,e2;hipEventCreate(&e1);hipEventCreate(&e2);
        hipEventRecord(e1,0);
        for(int i=0;i<it;++i){hipMemset(cnt,0,4);k_atomic<<<1024,32>>>(dC,dA,dB,LO,cnt);}
        hipDeviceSynchronize();hipEventRecord(e2,0);hipEventSynchronize(e2);
        float ms;hipEventElapsedTime(&ms,e1,e2);
        hipEventDestroy(e1);hipEventDestroy(e2);
        vals.push_back(O*1024*16*LO/(ms/it*1e-3)/1e12);
    }
    Stats st;st.n=ns;std::sort(vals.begin(),vals.end());
    st.min=vals.front();st.max=vals.back();
    double sum=0;for(double v:vals)sum+=v;st.mean=sum/ns;
    double s2=0;for(double v:vals)s2+=(v-st.mean)*(v-st.mean);st.std=sqrt(s2/ns);
    return st;
}

template<int PHASE>
Stats sample_phase(int32_t* dC,int32_t* dA,int32_t* dB,int it,int ns){
    std::vector<double> vals;
    for(int s=0;s<ns;++s){
        k_phase<PHASE><<<1024,32>>>(dC,dA,dB,LO);hipDeviceSynchronize();
        hipEvent_t e1,e2;hipEventCreate(&e1);hipEventCreate(&e2);
        hipEventRecord(e1,0);
        for(int i=0;i<it;++i)k_phase<PHASE><<<1024,32>>>(dC,dA,dB,LO);
        hipDeviceSynchronize();hipEventRecord(e2,0);hipEventSynchronize(e2);
        float ms;hipEventElapsedTime(&ms,e1,e2);
        hipEventDestroy(e1);hipEventDestroy(e2);
        vals.push_back(O*1024*16*LO/(ms/it*1e-3)/1e12);
    }
    Stats st;st.n=ns;std::sort(vals.begin(),vals.end());
    st.min=vals.front();st.max=vals.back();
    double sum=0;for(double v:vals)sum+=v;st.mean=sum/ns;
    double s2=0;for(double v:vals)s2+=(v-st.mean)*(v-st.mean);st.std=sqrt(s2/ns);
    return st;
}

int main(){
    hipDeviceProp_t p;hipGetDeviceProperties(&p,0);
    printf("GPU: %s (%d CUs/64 SIMDs) @ 2780 MHz\n",p.name,p.multiProcessorCount*2);
    printf("═══ Deterministic Wave Staggering ═══\n\n");

    int32_t *dC,*dA,*dB,*cnt;
    hipMalloc(&dC,1024*16*8*4);hipMalloc(&dA,1024*2*4);
    hipMalloc(&dB,4096*4);hipMalloc(&cnt,4);
    std::vector<int32_t>hA(2048,0x32103210),hB(4096,0x76547654);
    hipMemcpy(dA,hA.data(),8192,hipMemcpyHostToDevice);
    hipMemcpy(dB,hB.data(),16384,hipMemcpyHostToDevice);

    // 30s warmup
    printf("Warmup (30s)...\n");
    for(int i=0;i<300;++i){hipMemset(cnt,0,4);k_atomic<<<1024,32>>>(dC,dA,dB,LO,cnt);}
    hipDeviceSynchronize();
    printf("Done.\n\n");

    int it=10, ns=20;
    printf("%-28s %8s %8s %8s %8s %8s\n","Kernel","Min","Max","Mean","Std","IPC");
    printf("%-28s %8s %8s %8s %8s %8s\n","------","---","---","----","---","---");

    // Baseline: atomic
    Stats at = sample_atomic(dC,dA,dB,cnt,it,ns);
    printf("%-28s %8.0f %8.0f %8.0f %8.0f %6.3f\n",
           "atomic (baseline)",at.min,at.max,at.mean,at.std,at.mean/5830);

    // Phase NOP: sync launch + deterministic delay
    for(int ph:{1,2,4,8,16,32}){
        Stats st;
        if(ph==1)st=sample_phase<1>(dC,dA,dB,it,ns);
        else if(ph==2)st=sample_phase<2>(dC,dA,dB,it,ns);
        else if(ph==4)st=sample_phase<4>(dC,dA,dB,it,ns);
        else if(ph==8)st=sample_phase<8>(dC,dA,dB,it,ns);
        else if(ph==16)st=sample_phase<16>(dC,dA,dB,it,ns);
        else st=sample_phase<32>(dC,dA,dB,it,ns);
        char buf[28];snprintf(buf,sizeof(buf),"phase=%d NOP (sync)",ph);
        printf("%-28s %8.0f %8.0f %8.0f %8.0f %6.3f\n",
               buf,st.min,st.max,st.mean,st.std,st.mean/5830);
    }

    printf("\n═══ Analysis ═══\n");
    printf("  atomic mean:       %.0f TOPs  (IPC=%.3f)\n",at.mean,at.mean/5830);
    printf("  atomic max:        %.0f TOPs  (IPC=%.3f) ← hardware capability\n",at.max,at.max/5830);
    printf("  atomic min:        %.0f TOPs  (IPC=%.3f)\n",at.min,at.min/5830);
    printf("  atomic std/mean:   %.1f%%\n",at.std/at.mean*100);
    printf("\n  If phase-NOP < atomic: NOP overhead degrades performance\n");
    printf("  If phase-NOP > atomic peak: staggering improved over random atomic\n");

    hipFree(dC);hipFree(dA);hipFree(dB);hipFree(cnt);
    return 0;
}
