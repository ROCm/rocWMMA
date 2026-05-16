#!/usr/bin/env python3
"""
DrMOS Gate Edge Time (t_r) — Harmonic Envelope Roll-off Method

Usage:
  python3 smps_edge_sweep.py <iq_file> [f_switch] [fs]
  e.g. python3 smps_edge_sweep.py gfx1200_10msps.iq 51050 10000000
"""
import numpy as np, sys, os

def read_iq_miri(path, max_samples=None):
    """Read miri_sdr 12-bit IQ (3 bytes per 2 samples)"""
    data = np.fromfile(path, dtype=np.uint8)
    n_pairs = len(data)//3
    if max_samples: n_pairs = min(n_pairs, max_samples)
    iq = np.zeros(n_pairs*2, dtype=np.float32)
    for j in range(0, n_pairs, 4):  # process in chunks of 4 for speed
        end = min(j+4, n_pairs)
        for k in range(j, end):
            b0,b1,b2 = int(data[k*3]),int(data[k*3+1]),int(data[k*3+2])
            iv = ((b1&0x0F)<<8)|b0
            qv = (b2<<4)|((b1&0xF0)>>4)
            if iv>=2048: iv-=4096
            if qv>=2048: qv-=4096
            iq[k*2]=iv/2048.0
            iq[k*2+1]=qv/2048.0
    return iq

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    path = sys.argv[1]
    f0 = float(sys.argv[2]) if len(sys.argv)>2 else 51050.0
    fs = float(sys.argv[3]) if len(sys.argv)>3 else 10e6
    max_h = int(sys.argv[4]) if len(sys.argv)>4 else 98  # 5MHz/51kHz≈98

    if not os.path.exists(path):
        print(f"ERROR: {path} not found"); sys.exit(1)

    print(f"Loading: {path}")
    iq = read_iq_miri(path, max_samples=80_000_000)  # ~8 sec at 10MSPS
    n_total = len(iq)//2
    print(f"  Samples: {n_total:,}  ({n_total/fs:.1f}s at {fs/1e6:.0f} MSPS)")
    print(f"  SMPS f0: {f0/1000:.2f} kHz  Max harmonic: {max_h}")

    # Use long FFT blocks for low noise floor
    nfft = 1 << 20  # 1M points → ~9.5 Hz bins at 10MSPS
    window = np.hanning(nfft)
    n_segs = n_total // nfft
    print(f"  FFT: {nfft:,} pts ({nfft/fs*1000:.1f}ms/block) × {n_segs} blocks")
    print(f"  RBW: {fs/nfft:.1f} Hz")

    # Average power spectrum
    spec_sum = np.zeros(nfft//2+1, dtype=np.float64)
    for s in range(n_segs):
        start = s * nfft
        chunk = iq[::2][start:start+nfft] + 1j*iq[1::2][start:start+nfft]
        chunk = chunk - np.mean(chunk)  # DC removal
        chunk *= window
        spec_sum += np.abs(np.fft.rfft(chunk))**2  # power, not voltage
        if s % 10 == 0: print(f"    block {s}/{n_segs}...")
    spec_avg = spec_sum / n_segs
    freqs = np.fft.rfftfreq(nfft, 1/fs)
    bin_w = freqs[1]-freqs[0]

    # Voltage spectrum from power
    volt = np.sqrt(spec_avg)
    # Estimate noise floor from high-frequency region (>4.5MHz)
    noise_mask = freqs > 4.5e6
    noise_floor = float(np.mean(volt[noise_mask])) if np.any(noise_mask) else float(np.mean(volt[-1000:]))
    print(f"  Noise floor: {20*np.log10(noise_floor+1e-30):.1f} dB")

    # Extract harmonic peaks
    print(f"\n  Extracting harmonics of {f0/1000:.2f} kHz...")
    harmonics = []
    for n in range(1, max_h+1):
        target = n * f0
        if target > freqs[-1] - 200: break

        idx = np.argmin(np.abs(freqs-target))
        # Search ±5 bins for the actual peak
        sr = 5
        lo, hi = max(0,idx-sr), min(len(volt)-1,idx+sr)
        peak = float(np.max(volt[lo:hi+1]))
        peak_idx = lo + np.argmax(volt[lo:hi+1])
        peak_freq = freqs[peak_idx]

        # Local noise: median of nearby bins excluding peak
        n_lo = max(0, idx-50)
        n_hi = min(len(volt)-1, idx+50)
        local_volt = volt[n_lo:n_hi+1]
        local_noise = float(np.median(local_volt))

        snr = peak / (local_noise + 1e-30)
        harmonics.append({
            'n': n, 'freq': peak_freq, 'peak': peak,
            'noise': local_noise, 'snr': snr
        })

    # Filter to reliable harmonics (SNR > 1.5 → 3.5 dB)
    valid = [h for h in harmonics if h['snr'] > 1.5]
    print(f"  Valid harmonics (SNR>3.5dB): {len(valid)} of {len(harmonics)}")

    if len(valid) < 6:
        print("\n  NOT ENOUGH HARMONICS ABOVE NOISE")
        print("  → t_r is below measurement limit at this sample rate")
        f_max = freqs[-1]
        t_r_ub = 1.0/(np.pi*f_max)*1e9
        print(f"  → t_r < {t_r_ub:.1f} ns (upper bound at {fs/1e6:.0f} MSPS)")
        return

    # Compute current envelope: I(f) ∝ V(f)/f
    f_arr = np.array([h['freq'] for h in valid])
    v_arr = np.array([h['peak'] for h in valid])
    n_arr = np.array([h['n'] for h in valid])

    with np.errstate(divide='ignore'):
        i_rel = v_arr / f_arr
    i_dB = 20 * np.log10(i_rel + 1e-30)
    log_f = np.log10(f_arr)

    # ============================================================
    # Two-line fit: flat below f_c, -20 dB/dec above f_c
    # Sweep f_c and find best fit
    # ============================================================
    best_err = 1e30; best_fc = None; best_t_r = None

    # Candidate corner frequencies: every harmonic transition
    for split_idx in range(4, len(valid)-3):
        f_candidate = f_arr[split_idx]

        # Low group: fit constant
        low_y = i_dB[:split_idx]
        low_mean = np.mean(low_y)

        # High group: fit -20 dB/dec from f_candidate
        high_log_f = log_f[split_idx:]
        high_y = i_dB[split_idx:]
        # Model: y = low_mean - 20*(log_f - log_fc) for f > fc
        high_pred = low_mean - 20 * (high_log_f - np.log10(f_candidate))

        err = np.sum((high_y - high_pred)**2) + np.sum((low_y - low_mean)**2)
        err /= len(valid)

        if err < best_err:
            best_err = err
            best_fc = f_candidate
            best_tr = 1.0/(np.pi*best_fc)

    f_c, t_r = best_fc, best_tr

    # ============================================================
    # Results
    # ============================================================
    print(f"\n{'='*65}")
    print(f"  DrMOS EDGE TIME EXTRACTION")
    print(f"{'='*65}")
    print(f"  GPU:         gfx1200 (RDNA4, TSMC 4nm)")
    print(f"  SMPS f0:     {f0/1000:.2f} kHz ({len(valid)} harmonics valid)")
    print(f"  f_c (corner): {f_c/1000:.2f} kHz")
    print(f"  t_r (edge):   {t_r*1e9:.2f} ns")
    print(f"  Fit error:    {best_err:.3f}")
    print(f"  Method:       -20 dB/dec harmonic envelope roll-off")

    # Quality assessment
    if len(valid) > 30 and best_err < 2.0:
        conf = "HIGH — many harmonics, clean roll-off"
    elif len(valid) > 15:
        conf = "MEDIUM — adequate harmonics for fit"
    else:
        conf = "LOW — few harmonics, t_r may be approaching measurement limit"
    print(f"  Confidence:   {conf}")

    # ============================================================
    # Harmonic table
    # ============================================================
    print(f"\n  Harmonic envelope:")
    print(f"  {'n':>4s} {'f(kHz)':>9s} {'V(f)':>8s} {'I(f)':>8s} {'I(dB)':>7s} {'SNR':>6s}")
    print(f"  {'-'*4} {'-'*9} {'-'*8} {'-'*8} {'-'*7} {'-'*6}")
    for h in valid:
        i_val = h['peak']/h['freq']
        i_db = 20*np.log10(i_val+1e-30)
        snr_db = 20*np.log10(h['snr'])
        marker = ' ← fc' if abs(h['freq']-f_c)/f_c < 0.1 else ''
        print(f"  {h['n']:4d} {h['freq']/1000:9.2f} {h['peak']:8.4f} {i_val:8.4f} {i_db:6.1f}dB {snr_db:5.1f}dB{marker}")

    # Asymptotes
    low_mean = np.mean(i_dB[f_arr < f_c])
    print(f"\n  Asymptotes:")
    print(f"    Low:  I(f) ≈ {low_mean:.1f} dB  (flat)")
    print(f"    High: I(f) ≈ {low_mean:.1f} - 20·log10(f/{f_c/1000:.1f}kHz) dB")
    print(f"    Corner: {f_c/1000:.2f} kHz → t_r = {t_r*1e9:.2f} ns")
    print(f"    di/dt ∝ 1/t_r ≈ {1/t_r*1e-9:.1f} ns⁻¹")

    print(f"\n  ✅ Done.")

if __name__ == "__main__":
    main()
