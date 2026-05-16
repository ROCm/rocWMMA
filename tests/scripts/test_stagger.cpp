// test_stagger.cpp — Prove k1 vs k2 gap root cause
//
// Two tests:
// 1. k2_noatomic: k2 without WQ atomic → distributed by blockIdx.x (like k1)
//    If this drops to k1 levels → serialization IS the key
// 2. k1_atomic: k1 but waves claim work via atomic (like k2)
//    If this rises to k2 levels → serialization IS the key
//
// Build:
//   /opt/llvm-amd/bin/clang++ -x hip --offload-arch=gfx1200 \
//     -I/home/yanli/work/ROCm/rocWMMA/library/include \
//     -I/opt/rocm/include -DROCWMMA_WAVE32_MODE=1 -O3 \
//     -L/opt/rocm/lib -lamdhip64 -o test_stagger test_stagger.cpp

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
// K0: k1 ORIGINAL (sync start baseline)
// ======================================================================
template<int CH, int TI>
__global__ __launch_bounds__(32,(CH*8+7+TI*4<=128)?2:1)
void k0(int32_t*C,const int32_t*A,const int32_t*B,int L){
    int w=blockIdx.x;int32_t bt[TI][4];
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
// K2_ATOMIC: k0 + atomic work-claim (k2's serialization in k1's structure)
//   Waves claim work via atomicAdd on a counter.
//   Same SWMMAC loop as k1, but waves are staggered by atomic contention.
// ======================================================================
__global__ __launch_bounds__(32,2)
void k2_atomic(int32_t*C,const int32_t*A,const int32_t*B,
               int L,int* counter){
    int claimed = atomicAdd(counter, 1);  // ← serialization point
    if(claimed >= TT) return;            // only TT waves do work
    int w = claimed;                     // work assigned by claim order

    int32_t bt[1][4];
    for(int t=0;t<1;++t)for(int j=0;j<4;++j)bt[t][j]=B[(w*1+t)*4+j];
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
        for(int j=0;j<8;++j)C[((w*1+t)*16+cc)*8+j]=ac[cc][j];
}

// ======================================================================
// K3_DIRECT: k2 WITHOUT atomic (direct dispatch by blockIdx.x)
//   Uses WQ-like struct but distributed without atomic.
//   If this drops to k1 levels → atomic serialization is the key.
// ======================================================================
enum WT:int32_t{WC=0};struct WI{WT t;int wid,ti,LL;int32_t*dA,*dB,*dC;};
struct WQ{static constexpr int CAP=2048;WI items[CAP];int hd,count,proc;};

template<int TI>
__global__ __launch_bounds__(32,2)
void k3_direct(WQ*q,int32_t*pA,int32_t*pB,int32_t*pC){
    // DIRECT dispatch — NO atomic, just use blockIdx.x
    int idx = blockIdx.x;
    if(idx >= q->count) return;
    WI item = q->items[idx];

    if(item.t==WC){
        int32_t bt[TI][4];
        for(int t=0;t<TI;++t)for(int j=0;j<4;++j)bt[t][j]=item.dB[(item.wid*TI+t)*4+j];
        alignas(32)int32_t ac[16][8]={};
        auto&ra=*reinterpret_cast<const rocwmma::SwmmacARegsT*>(item.dA+item.wid*2);
        for(int t=0;t<TI;++t){
            auto&rb=*reinterpret_cast<const rocwmma::SwmmacBRegsT*>(bt[t]);
            for(int i=0;i<item.LL;++i){
                #pragma unroll
                for(int cc=0;cc<16;++cc){
                    auto&rc=*reinterpret_cast<rocwmma::SwmmacAccumT*>(ac[cc]);
                    rc=rocwmma::SwmmacI4::exec(ra,rb,rc,0);
                }
            }
        }
        for(int t=0;t<TI;++t)for(int cc=0;cc<16;++cc)
            for(int j=0;j<8;++j)item.dC[((item.wid*TI+t)*16+cc)*8+j]=ac[cc][j];
    }
}

// ======================================================================
// K4_ORIG: original k2 WQ (reference)
// ======================================================================
template<int TI>
__global__ __launch_bounds__(32,2)
void k4_orig(WQ*q,int32_t*pA,int32_t*pB,int32_t*pC){
    int idx=atomicAdd(&q->hd,1);WI item;bool ok=false;
    if(idx<q->count){idx%=WQ::CAP;item=q->items[idx];ok=true;atomicAdd(&q->proc,1);}
    else atomicSub(&q->hd,1);
    if(!ok)return;
    if(item.t==WC){
        int32_t bt[TI][4];
        for(int t=0;t<TI;++t)for(int j=0;j<4;++j)bt[t][j]=item.dB[(item.wid*TI+t)*4+j];
        alignas(32)int32_t ac[16][8]={};
        auto&ra=*reinterpret_cast<const rocwmma::SwmmacARegsT*>(item.dA+item.wid*2);
        for(int t=0;t<TI;++t){
            auto&rb=*reinterpret_cast<const rocwmma::SwmmacBRegsT*>(bt[t]);
            for(int i=0;i<item.LL;++i){
                #pragma unroll
                for(int cc=0;cc<16;++cc){
                    auto&rc=*reinterpret_cast<rocwmma::SwmmacAccumT*>(ac[cc]);
                    rc=rocwmma::SwmmacI4::exec(ra,rb,rc,0);
                }
            }
        }
        for(int t=0;t<TI;++t)for(int cc=0;cc<16;++cc)
            for(int j=0;j<8;++j)item.dC[((item.wid*TI+t)*16+cc)*8+j]=ac[cc][j];
    }
}

// ======================================================================
// Bench helpers
// ======================================================================
double bk0(int w,int L,int it){
    int32_t*dC,*dA,*dB;hipMalloc(&dC,w*16*8*4);hipMalloc(&dA,w*2*4);hipMalloc(&dB,w*4*4);
    std::vector<int32_t>hA(w*2,0x32103210),hB(w*4,0x76547654);
    hipMemcpy(dA,hA.data(),w*8,hipMemcpyHostToDevice);hipMemcpy(dB,hB.data(),w*16,hipMemcpyHostToDevice);
    for(int i=0;i<3;++i)k0<16,1><<<w,32>>>(dC,dA,dB,L);hipDeviceSynchronize();
    hipEvent_t s,e;hipEventCreate(&s);hipEventCreate(&e);hipEventRecord(s,0);
    for(int i=0;i<it;++i)k0<16,1><<<w,32>>>(dC,dA,dB,L);
    hipEventRecord(e,0);hipEventSynchronize(e);float ms;hipEventElapsedTime(&ms,s,e);
    hipEventDestroy(s);hipEventDestroy(e);hipFree(dC);hipFree(dA);hipFree(dB);
    return O*w*16*L/(ms/it*1e-3)/1e12;
}

double bk2(int w,int L,int it){
    int32_t*dC,*dA,*dB,*cnt;hipMalloc(&dC,w*16*8*4);hipMalloc(&dA,w*2*4);hipMalloc(&dB,w*4*4);
    hipMalloc(&cnt,4);hipMemset(cnt,0,4);
    std::vector<int32_t>hA(w*2,0x32103210),hB(w*4,0x76547654);
    hipMemcpy(dA,hA.data(),w*8,hipMemcpyHostToDevice);hipMemcpy(dB,hB.data(),w*16,hipMemcpyHostToDevice);
    for(int i=0;i<3;++i){hipMemset(cnt,0,4);k2_atomic<<<w*2,32>>>(dC,dA,dB,L,cnt);}hipDeviceSynchronize();
    hipEvent_t s,e;hipEventCreate(&s);hipEventCreate(&e);hipEventRecord(s,0);
    for(int i=0;i<it;++i){hipMemset(cnt,0,4);k2_atomic<<<w*2,32>>>(dC,dA,dB,L,cnt);}hipDeviceSynchronize();
    hipEventRecord(e,0);hipEventSynchronize(e);float ms;hipEventElapsedTime(&ms,s,e);
    hipEventDestroy(s);hipEventDestroy(e);hipFree(dC);hipFree(dA);hipFree(dB);hipFree(cnt);
    return O*w*16*L/(ms/it*1e-3)/1e12;
}

template<int TI>
double bk3(int workers,int L,int it){
    int tiles=TI,tt=workers*tiles;int32_t*dC,*dA,*dB;
    hipMalloc(&dC,tt*16*8*4);hipMalloc(&dA,workers*2*4);hipMalloc(&dB,tt*4*4);
    std::vector<int32_t>hA(workers*2,0x32103210),hB(tt*4,0x76547654);
    hipMemcpy(dA,hA.data(),workers*8,hipMemcpyHostToDevice);hipMemcpy(dB,hB.data(),tt*16,hipMemcpyHostToDevice);
    WQ hq={};for(int i=0;i<workers;++i){hq.items[i].t=WC;hq.items[i].wid=i;hq.items[i].ti=tiles;hq.items[i].LL=L;hq.items[i].dA=dA;hq.items[i].dB=dB;hq.items[i].dC=dC;}hq.count=workers;
    WQ*dq;hipMalloc(&dq,sizeof(WQ));
    hipMemcpy(dq,&hq,sizeof(WQ),hipMemcpyHostToDevice);
    for(int w=0;w<3;++w)k3_direct<TI><<<workers,32>>>(dq,dA,dB,dC);hipDeviceSynchronize();
    hipEvent_t s,e;hipEventCreate(&s);hipEventCreate(&e);hipEventRecord(s,0);
    for(int i=0;i<it;++i)k3_direct<TI><<<workers,32>>>(dq,dA,dB,dC);
    hipEventRecord(e,0);hipEventSynchronize(e);float ms;hipEventElapsedTime(&ms,s,e);
    hipEventDestroy(s);hipEventDestroy(e);hipFree(dC);hipFree(dA);hipFree(dB);hipFree(dq);
    return O*workers*16*tiles*L/(ms/it*1e-3)/1e12;
}

template<int TI>
double bk4(int workers,int L,int it){
    int tiles=TI,tt=workers*tiles;int32_t*dC,*dA,*dB;
    hipMalloc(&dC,tt*16*8*4);hipMalloc(&dA,workers*2*4);hipMalloc(&dB,tt*4*4);
    std::vector<int32_t>hA(workers*2,0x32103210),hB(tt*4,0x76547654);
    hipMemcpy(dA,hA.data(),workers*8,hipMemcpyHostToDevice);hipMemcpy(dB,hB.data(),tt*16,hipMemcpyHostToDevice);
    WQ hq={};for(int i=0;i<workers;++i){hq.items[i].t=WC;hq.items[i].wid=i;hq.items[i].ti=tiles;hq.items[i].LL=L;hq.items[i].dA=dA;hq.items[i].dB=dB;hq.items[i].dC=dC;}hq.count=workers;
    WQ*dq;hipMalloc(&dq,sizeof(WQ));
    for(int w=0;w<3;++w){hipMemcpy(dq,&hq,sizeof(WQ),hipMemcpyHostToDevice);k4_orig<TI><<<workers,32>>>(dq,dA,dB,dC);}hipDeviceSynchronize();
    hipEvent_t s,e;hipEventCreate(&s);hipEventCreate(&e);hipEventRecord(s,0);
    for(int i=0;i<it;++i){hipMemcpy(dq,&hq,sizeof(WQ),hipMemcpyHostToDevice);k4_orig<TI><<<workers,32>>>(dq,dA,dB,dC);}hipDeviceSynchronize();
    hipEventRecord(e,0);hipEventSynchronize(e);float ms;hipEventElapsedTime(&ms,s,e);
    hipEventDestroy(s);hipEventDestroy(e);hipFree(dC);hipFree(dA);hipFree(dB);hipFree(dq);
    return O*workers*16*tiles*L/(ms/it*1e-3)/1e12;
}

int main(){
    hipDeviceProp_t p;hipGetDeviceProperties(&p,0);int nCu=p.multiProcessorCount*2;
    printf("GPU: %s (%d CUs, %d SIMDs)\n\n",p.name,nCu,nCu*2);
    int W=TT, L=LO, it=15;

    printf("═══ Root Cause Analysis: k1 vs k2 gap ═══\n\n");
    printf("%-30s %8s  %s\n","Kernel","TOPs","Property");
    printf("%-30s %8s  %s\n","------","----","--------");

    // 1. k1 ORIGINAL (sync baseline)
    double k1tp = bk0(W,L,it);
    printf("%-30s %8.0f  %s\n","K1_SYNC (blockIdx.x)",k1tp,"sync start");

    // 2. k1 + ATOMIC work-claim (staggered)
    double k2tp = bk2(W,L,it);
    printf("%-30s %8.0f  %s\n","K1+ATOMIC (work-claim)",k2tp,"staggered via atomic");

    // 3. k2 WQ WITHOUT atomic (direct dispatch)
    double k3tp = bk3<1>(W,L,it);
    printf("%-30s %8.0f  %s\n","K2_WQ_NOATOMIC (direct)",k3tp,"no stagger");

    // 4. k2 WQ ORIGINAL (atomic dispatch - reference)
    double k4tp = bk4<1>(W,L,it);
    printf("%-30s %8.0f  %s\n","K2_WQ_ATOMIC (orig)",k4tp,"staggered via atomic");

    printf("\n═══ Analysis ═══\n");
    printf("k1 sync → k1+atomic gap:  %.0f%%  ",(k2tp/k1tp-1)*100);
    if(k2tp > k1tp*1.2) printf("STAGGER CONFIRMED (+>20%%)\n");
    else printf("no significant change\n");

    printf("k2 direct → k2 atomic gap: %.0f%%  ",(k4tp/k3tp-1)*100);
    if(k4tp > k3tp*1.2) printf("STAGGER CONFIRMED (+>20%%)\n");
    else printf("no significant change\n");

    printf("k1 sync vs k2 direct:      %.0f%%  ",(k3tp/k1tp-1)*100);
    if(k3tp <= k1tp*1.05) printf("IDENTICAL → stagger is the ROOT CAUSE\n");
    else printf("other difference exists\n");

    return 0;
}
