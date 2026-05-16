// bench_persistent_counter.cpp — Eliminate counter reset, let it wrap
//
// Every hipMemset(cnt,0,4) evicts the counter from L2, guaranteeing a cold miss.
// Solution: use a SEQUENCE NUMBER instead of zeroing the counter.
//   1. Host maintains a "base" sequence value
//   2. Kernel does: claimed = atomicAdd(cnt, 1) - base; if(claimed >= TT) return;
//   3. After each launch: base += TT (no need to reset cnt to zero)
//
// Build: /opt/llvm-amd/bin/clang++ -x hip --offload-arch=gfx1200 \
//   -I/opt/rocm/include -DROCWMMA_WAVE32_MODE=1 -O3 \
//   -L/opt/rocm/lib -lamdhip64 -o bench_persistent_counter bench_persistent_counter.cpp

#include <rocwmma/rocwmma_16chain.hpp>
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include <ctime>
static constexpr double O=32768.0;
static constexpr int LO=160, TT=1024;
static constexpr double TH=5306.0;

// k6: ALL 32 threads do atomicAdd — 32× contention IS the stagger mechanism
// Wrap mode: base accounts for 32*TT claims per launch (32 threads × TT waves)
__global__ __launch_bounds__(32,2)
void k6(int32_t*C,const int32_t*A,const int32_t*B,int L,int* cnt,int base){
    int claimed=atomicAdd(cnt,1);
    if(claimed-base>=TT)return;
    int w=claimed-base;
    int32_t bt[4];for(int j=0;j<4;++j)bt[j]=B[w*4+j];
    alignas(32)int32_t ac[16][8]={};
    const rocwmma::SwmmacARegsT& ra=*reinterpret_cast<const rocwmma::SwmmacARegsT*>(A+w*2);
    const rocwmma::SwmmacBRegsT& rb=*reinterpret_cast<const rocwmma::SwmmacBRegsT*>(bt);
    for(int i=0;i<L;++i){
        #pragma unroll
        for(int cc=0;cc<16;++cc){
            rocwmma::SwmmacAccumT& rc=*reinterpret_cast<rocwmma::SwmmacAccumT*>(ac[cc]);rc=rocwmma::SwmmacI4::exec(ra,rb,rc,0);}
    }
    for(int cc=0;cc<16;++cc)for(int j=0;j<8;++j)C[(w*16+cc)*8+j]=ac[cc][j];
}

// Simple counter-based benchmark with and without hipMemset
double bench_reset(int32_t*dC,int32_t*dA,int32_t*dB,int32_t*cnt,int it,int ns){
    std::vector<double>v;
    for(int s=0;s<ns;++s){
        hipMemset(cnt,0,4);  // RESET — evicts counter from L2
        k6<<<1024,32>>>(dC,dA,dB,LO,cnt,0);
        hipDeviceSynchronize();
        hipEvent_t e1,e2;hipEventCreate(&e1);hipEventCreate(&e2);
        hipEventRecord(e1,0);
        for(int i=0;i<it;++i){hipMemset(cnt,0,4);k6<<<1024,32>>>(dC,dA,dB,LO,cnt,0);}
        hipDeviceSynchronize();hipEventRecord(e2,0);hipEventSynchronize(e2);
        float ms;hipEventElapsedTime(&ms,e1,e2);
        hipEventDestroy(e1);hipEventDestroy(e2);
        v.push_back(O*1024*16*LO/(ms/it*1e-3)/1e12);
    }
    std::sort(v.begin(),v.end());
    double sum=0;for(double x:v)sum+=x;
    return sum/v.size();
}

double bench_wrap(int32_t*dC,int32_t*dA,int32_t*dB,int32_t*cnt,int it,int ns){
    // Initialize counter once, never reset
    hipMemset(cnt,0,4);
    int base=0;
    std::vector<double>v;
    for(int s=0;s<ns;++s){
        // NO hipMemset! Counter stays in L2 between launches
        k6<<<1024,32>>>(dC,dA,dB,LO,cnt,base);
        hipDeviceSynchronize();
        int per_launch=32*TT;  // 32 threads × 1024 blocks
        base+=per_launch;      // advance base for next launch

        hipEvent_t e1,e2;hipEventCreate(&e1);hipEventCreate(&e2);
        hipEventRecord(e1,0);
        for(int i=0;i<it;++i){
            k6<<<1024,32>>>(dC,dA,dB,LO,cnt,base+i*per_launch);
        }
        hipDeviceSynchronize();hipEventRecord(e2,0);hipEventSynchronize(e2);
        float ms;hipEventElapsedTime(&ms,e1,e2);
        hipEventDestroy(e1);hipEventDestroy(e2);
        v.push_back(O*1024*16*LO/(ms/it*1e-3)/1e12);
        base+=it*per_launch;
    }
    std::sort(v.begin(),v.end());
    double sum=0;for(double x:v)sum+=x;
    return sum/v.size();
}

int main(){
    hipDeviceProp_t p;hipGetDeviceProperties(&p,0);
    printf("═══ Persistent Counter: Eliminate hipMemset Eviction ═══\n");
    printf("GPU: %s @ %.0f MHz  Theory: %.0f TOPs\n\n",
           p.name,p.clockRate/1000.0,TH);

    int32_t *dC,*dA,*dB,*cnt;
    hipMalloc(&dC,1024*16*8*4);hipMalloc(&dA,1024*2*4);
    hipMalloc(&dB,4096*4);hipMalloc(&cnt,4);
    std::vector<int32_t>hA(2048,0x32103210),hB(4096,0x76547654);
    hipMemcpy(dA,hA.data(),8192,hipMemcpyHostToDevice);
    hipMemcpy(dB,hB.data(),16384,hipMemcpyHostToDevice);

    printf("Thermal warmup (60s)...\n");
    time_t t0=time(0);
    while(time(0)-t0<60){hipMemset(cnt,0,4);k6<<<1024,32>>>(dC,dA,dB,LO,cnt,0);}
    hipDeviceSynchronize();
    printf("Done.\n\n");

    int it=10, ns=60;
    printf("Phase 1: With hipMemset reset (baseline)...\n");
    double r_mean=bench_reset(dC,dA,dB,cnt,it,ns);
    printf("Phase 2: Wrap counter (no reset, L2 persistent)...\n");
    double w_mean=bench_wrap(dC,dA,dB,cnt,it,ns);

    printf("\n═══ Results ═══\n");
    printf("  Reset (hipMemset): %.0f TOPs\n",r_mean);
    printf("  Wrap  (persistent): %.0f TOPs  (%+.1f%%)\n",w_mean,(w_mean/r_mean-1)*100);

    if(w_mean>r_mean) printf("\n  ✓ Persistent counter eliminates L2 eviction penalty\n");
    else printf("\n  ~ Counter L2 state not the bottleneck in multi-launch\n");

    hipFree(dC);hipFree(dA);hipFree(dB);hipFree(cnt);
    return 0;
}
