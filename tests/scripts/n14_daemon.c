/* n14_daemon.c — N14-locked SMPS Phase Monitor (daemon, miri 12-bit IQ)
 * Build: gcc -O3 -march=native -ffast-math -o n14_daemon n14_daemon.c -lm -lrt
 * Usage:
 *   ./n14_daemon --source-file /tmp/smps.iq --fs 15e6 --f0 51050 --foreground
 *   miri_sdr -f 0 -s 15e6 -g 3 - | ./n14_daemon --source-stdin --fs 15e6 --f0 51050
 *
 * HIP reader:
 *   int fd = shm_open("/n14_smps_fingerprint", O_RDONLY, 0);
 *   n14_shm_t *n14 = mmap(NULL, sizeof(n14_shm_t), PROT_READ, MAP_SHARED, fd, 0);
 *   if (n14->qtf_0_1_0_5_hz_ns > 30.0) launch_tuned_kernel();
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <complex.h>
#include <math.h>
#include <string.h>
#include <signal.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <errno.h>
#include <time.h>

#define BLOCK_SIZE       262144
#define DECIMATE         1000
#define UPDATE_INTERVAL  10.0
#define SHM_NAME         "/n14_smps_fingerprint"

typedef struct {
    double locked_freq;
    double freq_std;
    double rms_jitter_ns;
    double qtf_0_1_0_5_hz_ns;
    double rms_0_5_10_hz_ns;
    uint64_t total_samples;
    time_t last_update;
} n14_shm_t;

static volatile int keep_running = 1;
static n14_shm_t *shm_ptr = NULL;
static int shm_fd = -1;
static double f0_global = 51050.0;

void signal_handler(int sig) { keep_running = 0; }

void daemonize(void) {
    pid_t pid = fork(); if (pid < 0) exit(1); if (pid > 0) _exit(0);
    if (setsid() < 0) exit(1);
    signal(SIGCHLD, SIG_IGN); signal(SIGHUP, SIG_IGN);
    pid = fork(); if (pid < 0) exit(1); if (pid > 0) _exit(0);
    umask(0); chdir("/");
    for (int x = sysconf(_SC_OPEN_MAX); x >= 0; x--) close(x);
    int fd0 = open("/dev/null", O_RDWR); dup2(fd0, 0); dup2(fd0, 1); dup2(fd0, 2);
    if (fd0 > 2) close(fd0);
}

static inline int miri_decode(uint8_t *raw, float *iq, int n_pairs) {
    for (int j = 0; j < n_pairs; j++) {
        uint8_t b0=raw[j*3], b1=raw[j*3+1], b2=raw[j*3+2];
        int iv=((b1&0x0F)<<8)|b0, qv=(b2<<4)|((b1&0xF0)>>4);
        if(iv>=2048)iv-=4096; if(qv>=2048)qv-=4096;
        iq[j*2]=(float)iv/2048.0f; iq[j*2+1]=(float)qv/2048.0f;
    }
    return n_pairs;
}

void compute_band_rms(const float *phase_err, int len, double fs_phase,
                      double flo, double fhi, double *rms_ns) {
    double w0=2.0*M_PI*flo/fs_phase, w1=2.0*M_PI*fhi/fs_phase;
    double B=w1-w0, S=w0*w1, Q=sqrt(S)/(B+1e-15);
    double alpha=sin(B)/(2.0*Q+1e-15);
    double b0=alpha, b1=0.0, b2=-alpha;
    double a0_=1.0+alpha, a1=-2.0*cos((w0+w1)/2.0), a2=1.0-alpha;
    b0/=a0_; b1/=a0_; b2/=a0_; a1/=a0_; a2/=a0_;
    double x1=0,x2=0,y1=0,y2=0,sum_sq=0.0;
    for(int i=0;i<len;i++){double x0=phase_err[i];
        double y0=b0*x0+b1*x1+b2*x2-a1*y1-a2*y2;
        x2=x1;x1=x0;y2=y1;y1=y0;sum_sq+=y0*y0;}
    *rms_ns=sqrt(sum_sq/len)/(2.0*M_PI*f0_global)*1e9;
}

void update_shm(double freq,double fstd,double rms_jitter,double qtf,double rms2,uint64_t n){
    if(!shm_ptr)return;
    shm_ptr->locked_freq=freq;shm_ptr->freq_std=fstd;
    shm_ptr->rms_jitter_ns=rms_jitter;shm_ptr->qtf_0_1_0_5_hz_ns=qtf;
    shm_ptr->rms_0_5_10_hz_ns=rms2;shm_ptr->total_samples=n;
    shm_ptr->last_update=time(NULL);
}

int setup_shm(void){
    shm_fd=shm_open(SHM_NAME,O_CREAT|O_RDWR,0666);
    if(shm_fd<0){perror("shm_open");return -1;}
    if(ftruncate(shm_fd,sizeof(n14_shm_t))<0){perror("ftruncate");return -1;}
    shm_ptr=mmap(NULL,sizeof(n14_shm_t),PROT_READ|PROT_WRITE,MAP_SHARED,shm_fd,0);
    if(shm_ptr==MAP_FAILED){perror("mmap");return -1;}
    memset(shm_ptr,0,sizeof(n14_shm_t));
    return 0;
}

void process_stream(FILE *input, double fs, double f0, double loop_bw){
    double kp=2.0*M_PI*loop_bw/fs, ki=kp*kp/(4.0*0.707*0.707);
    double phase_est=0.0,freq_est=0.0; uint64_t ns=0;
    double fs_phase=fs/DECIMATE;
    int update_n=(int)(UPDATE_INTERVAL*fs_phase);
    float *ph=malloc(update_n*sizeof(float)); double *fh=malloc(update_n*sizeof(double));
    int pi=0,pc=0,fi=0,fc=0;
    size_t raw_sz = (size_t)BLOCK_SIZE * 3;    /* 3 bytes per sample pair */
    uint8_t *raw = malloc(raw_sz);
    float *iq = malloc((size_t)BLOCK_SIZE * 2 * sizeof(float));

    while(keep_running){
        size_t nr = fread(raw, 1, raw_sz, input);
        if(nr<3){if(ferror(input))break;usleep(10000);clearerr(input);continue;}
        int np=miri_decode(raw,iq,(int)nr/3);
        for(int i=0;i<np;i++){
            float complex sig=iq[i*2]+I*iq[i*2+1];
            float complex lo=cexpf(-I*(float)phase_est);
            float complex prod=sig*lo; float pd=cargf(prod);
            freq_est+=ki*pd; phase_est+=freq_est+kp*pd;
            phase_est=fmodf(phase_est,2.0f*(float)M_PI);
            if(ns%DECIMATE==0){ph[pi]=pd;pi=(pi+1)%update_n;if(pc<update_n)pc++;
                double inst_f=f0+freq_est*fs/(2.0*M_PI);
                fh[fi]=inst_f;fi=(fi+1)%update_n;if(fc<update_n)fc++;}
            ns++;
        }
        if(pc>=update_n){
            double fs_=0,fs2_=0; for(int i=0;i<fc;i++){fs_+=fh[i];fs2_+=fh[i]*fh[i];}
            double fm=fs_/fc, fstd=sqrt(fs2_/fc-fm*fm);
            double ps2=0; for(int i=0;i<pc;i++)ps2+=(double)ph[i]*ph[i];
            double rj=sqrt(ps2/pc)/(2.0*M_PI*f0)*1e9;
            double qtf,rb2; compute_band_rms(ph,pc,fs_phase,0.1,0.5,&qtf);
            compute_band_rms(ph,pc,fs_phase,0.5,10.0,&rb2);
            update_shm(fm,fstd,rj,qtf,rb2,ns);
            fprintf(stderr,"QTF=%.1fns f=%.4fHz σf=%.4fHz jitter=%.1fns samples=%lu\n",
                    qtf,fm,fstd,rj,(unsigned long)ns);
            pc=0;fc=0;pi=0;fi=0;
        }
    }
    free(raw);free(iq);free(ph);free(fh);
}

int main(int argc,char**argv){
    double fs=15e6,f0=51050.0,bw=0.5;
    char*sfile=NULL; int use_stdin=0,fg=0;
    for(int i=1;i<argc;i++){
        if(!strcmp(argv[i],"--fs")&&i+1<argc)fs=atof(argv[++i]);
        else if(!strcmp(argv[i],"--f0")&&i+1<argc)f0=atof(argv[++i]);
        else if(!strcmp(argv[i],"--bw")&&i+1<argc)bw=atof(argv[++i]);
        else if(!strcmp(argv[i],"--source-file")&&i+1<argc)sfile=argv[++i];
        else if(!strcmp(argv[i],"--source-stdin"))use_stdin=1;
        else if(!strcmp(argv[i],"--foreground"))fg=1;
        else{fprintf(stderr,"Usage: %s [--fs FS] [--f0 F0] [--bw BW] [--source-file <f>|--source-stdin] [--foreground]\n",argv[0]);return 1;}
    }
    if(!sfile&&!use_stdin){fprintf(stderr,"Need --source-file or --source-stdin\n");return 1;}
    f0_global=f0;
    if(setup_shm()<0)return 1;
    signal(SIGTERM,signal_handler);signal(SIGINT,signal_handler);
    if(!fg)daemonize();
    FILE*in=use_stdin?stdin:fopen(sfile,"rb");
    if(!in){perror("fopen");return 1;}
    if(!use_stdin)setvbuf(in,NULL,_IONBF,0);
    fprintf(stderr,"n14_daemon: fs=%.0f f0=%.1f bw=%.1f\n",fs,f0,bw);
    process_stream(in,fs,f0,bw);
    if(!use_stdin)fclose(in);
    munmap(shm_ptr,sizeof(n14_shm_t));shm_unlink(SHM_NAME);
    return 0;
}
