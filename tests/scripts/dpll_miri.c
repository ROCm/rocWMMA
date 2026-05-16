/* dpll_miri.c — N14-locked DPLL for miri_sdr 12-bit IQ
   Build: gcc -O3 -march=native -ffast-math -o /tmp/dpll_miri /tmp/dpll_miri.c -lm
   Usage: /tmp/dpll_miri <input.iq> <output.csv> <f0_hz> <loop_bw_hz> <fs_hz> <start_sec> <duration_sec>
*/
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <complex.h>
#include <math.h>
#include <string.h>

#define BLOCK_SIZE 262144
#define DECIMATE 1000

typedef float complex cmplx;

/* Decode miri 12-bit: 3 bytes → 2 samples */
static inline void miri_decode(uint8_t *raw, float *iq, int n_pairs) {
    for (int j = 0; j < n_pairs; j++) {
        uint8_t b0 = raw[j*3], b1 = raw[j*3+1], b2 = raw[j*3+2];
        int iv = ((b1 & 0x0F) << 8) | b0;
        int qv = (b2 << 4) | ((b1 & 0xF0) >> 4);
        if (iv >= 2048) iv -= 4096;
        if (qv >= 2048) qv -= 4096;
        iq[j*2]   = iv / 2048.0f;
        iq[j*2+1] = qv / 2048.0f;
    }
}

int main(int argc, char **argv) {
    if (argc != 8) {
        fprintf(stderr, "Usage: %s input.iq output.csv f0 loop_bw fs start_sec duration_sec\n", argv[0]);
        return 1;
    }
    const char *infile  = argv[1];
    const char *outfile = argv[2];
    double f0       = atof(argv[3]);
    double loop_bw  = atof(argv[4]);
    double fs       = atof(argv[5]);
    double start_sec = atof(argv[6]);
    double dur_sec  = atof(argv[7]);

    double kp = 2.0 * M_PI * loop_bw / fs;
    double ki = kp * kp / (4.0 * 0.707 * 0.707);

    double phase_est = 0.0, freq_est = 0.0;
    uint64_t total = 0;

    FILE *fin = fopen(infile, "rb");
    FILE *fout = fopen(outfile, "w");
    if (!fin || !fout) { perror("fopen"); return 1; }

    /* Seek to start position (3 bytes per 2 samples) */
    long skip_bytes = (long)(start_sec * fs * 1.5);
    fseek(fin, skip_bytes, SEEK_SET);

    long max_bytes = (long)(dur_sec * fs * 1.5);
    long bytes_read = 0;

    uint8_t *raw = malloc(BLOCK_SIZE * 3 / 2 * 3); /* enough for BLOCK_SIZE/2 pairs */
    float *iq  = malloc(BLOCK_SIZE * sizeof(float));     /* interleaved I/Q float32 */

    fprintf(fout, "sample,phase_err,inst_freq_hz\n");
    fprintf(stderr, "DPLL start: f0=%.1f bw=%.1f fs=%.0f start=%.1f dur=%.1f\n",
            f0, loop_bw, fs, start_sec, dur_sec);

    while (bytes_read < max_bytes) {
        int to_read = (int)fmin(BLOCK_SIZE * 3 / 2, max_bytes - bytes_read);
        size_t n = fread(raw, 1, to_read, fin);
        if (n < 3) break;
        int n_pairs = (int)n / 3;
        bytes_read += n;

        miri_decode(raw, iq, n_pairs);

        for (int i = 0; i < n_pairs; i++) {
            cmplx sig = iq[i*2] + I * iq[i*2+1];
            cmplx lo = cexpf(-I * (float)phase_est);
            cmplx prod = sig * lo;
            float ph_det = cargf(prod);

            freq_est += ki * ph_det;
            phase_est += freq_est + kp * ph_det;
            phase_est = fmodf(phase_est, (float)(2.0 * M_PI));

            if (total % DECIMATE == 0) {
                double inst_freq = f0 + freq_est * fs / (2.0 * M_PI);
                fprintf(fout, "%lu,%f,%f\n", (unsigned long)total, ph_det, inst_freq);
            }
            total++;
        }

        if (total % 50000000 == 0)
            fprintf(stderr, "  %.1fs processed (%lu samples, %.1fs elapsed)\n",
                    (double)total / fs, (unsigned long)total, bytes_read / (fs * 1.5));
    }

    fclose(fin); fclose(fout);
    free(raw); free(iq);

    /* Summary */
    fprintf(stderr, "\nDone. Total: %lu samples (%.2fs). Output: %s\n",
            (unsigned long)total, (double)total / fs, outfile);
    return 0;
}
