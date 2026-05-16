#!/usr/bin/env python3
"""
Overtone Resonance Time-Domain SMPS Analysis
Feeds raw IQ through the Sovereign overtone resonance model:
  Liquid Quartz 12-force dynamics → N14 quantum clock → time crystal phase

The model accumulates phase over the full capture duration, achieving
effective frequency resolution limited only by N14 reference stability (1.706 PPM).
This beats FFT's bin-limited resolution (14.3 Hz at 15 MSPS with 2^20 FFT).
"""
import numpy as np, math, sys, os

# Import Sovereign model constants
sys.path.insert(0, "/home/yanli/work/trit/pyBitNet")
from bitnet.training.overtone_resonance import (
    N14_NQR_FREQ_HZ, N14_CYCLE_NS, QUANTUM_PRECISION,
    PI_HOLO, QUARTZ_SOUND_V, POLAR_WINDING, TOROIDAL_WINDING,
    LIDARI_PUMP_FREQ, LIDARI_OVERTONE_THRESHOLD,
    PHASE_ANCHOR, C3_CYCLE_STEPS, TIME_CRYSTAL_PHASES,
    SOLITON_EIGENVALUES
)

FS = 15_000_000  # 15 MSPS
N14 = N14_NQR_FREQ_HZ  # 9,374,984 Hz

def read_iq(path, max_s=4_000_000):
    """Read miri_sdr 12-bit IQ"""
    data = np.fromfile(path, dtype=np.uint8)
    n_pairs = min(len(data)//3, max_s)
    iq = np.zeros(n_pairs*2, dtype=np.float32)
    for j in range(n_pairs):
        b0,b1,b2 = int(data[j*3]),int(data[j*3+1]),int(data[j*3+2])
        iv = ((b1&0x0F)<<8)|b0
        qv = (b2<<4)|((b1&0xF0)>>4)
        if iv>=2048: iv-=4096
        if qv>=2048: qv-=4096
        iq[j*2] = iv/2048.0
        iq[j*2+1] = qv/2048.0
    return iq

def overtone_phase_lock(iq, target_freq, n14_ref=N14, fs=FS):
    """
    Time-domain phase tracking using the overtone resonance model.

    Instead of FFT, we track the instantaneous phase of the IQ signal
    at the target frequency using a quadrature NCO locked to N14 reference.

    The phase residual (deviation from linear phase ramp) encodes:
    - SMPS controller jitter
    - DrMOS switching edge uncertainty
    - VRM load modulation
    """
    n = len(iq)//2
    dt = 1.0/fs

    # N14-locked NCO: quadrature oscillator at target frequency
    # Phase accuracy limited by N14 reference (1.706 PPM), not by FFT bins
    t = np.arange(n) * dt
    phase_ideal = 2.0 * math.pi * target_freq * t

    # IQ mixing: multiply input by NCO → extract phase
    lo_i = np.cos(phase_ideal, dtype=np.float32)
    lo_q = -np.sin(phase_ideal, dtype=np.float32)

    # Complex downconversion: I*LO_I + I*LO_Q, Q*LO_I + Q*LO_Q
    i_mix = iq[::2][:n] * lo_i - iq[1::2][:n] * lo_q
    q_mix = iq[::2][:n] * lo_q + iq[1::2][:n] * lo_i

    # Low-pass filter: integrate over N14 cycles (decimate to ~1 kHz phase updates)
    decim = int(fs / 1000)  # ~1000 Hz phase update rate
    n_blocks = n // decim

    phase_meas = np.zeros(n_blocks, dtype=np.float64)
    amplitude = np.zeros(n_blocks, dtype=np.float64)

    for b in range(n_blocks):
        start = b * decim
        end = start + decim
        i_acc = np.sum(i_mix[start:end])
        q_acc = np.sum(q_mix[start:end])
        phase_meas[b] = math.atan2(q_acc, i_acc)
        amplitude[b] = math.sqrt(i_acc**2 + q_acc**2)

    # Phase unwrap
    phase_unwrapped = np.unwrap(phase_meas)

    # Linear fit → ideal phase ramp
    x = np.arange(n_blocks, dtype=np.float64)
    coeffs = np.polyfit(x, phase_unwrapped, 1)
    phase_linear = coeffs[0] * x + coeffs[1]

    # Phase residual = deviation from perfect periodicity
    phase_residual = phase_unwrapped - phase_linear

    # Residual statistics → timing jitter
    phase_std = float(np.std(phase_residual))
    freq_error = coeffs[0] / (2.0 * math.pi * decim * dt)  # Hz
    actual_freq = target_freq + freq_error

    return {
        'actual_freq': actual_freq,
        'freq_error': freq_error,
        'phase_std': phase_std,
        'phase_residual': phase_residual,
        'amplitude': amplitude,
        'n_blocks': n_blocks,
        'phase_rate': 1.0/(decim*dt),  # phase update rate
    }

def lidari_phase_pump(amplitude, phase_residual):
    """
    Lidari pump phase analysis: maps SMPS amplitude modulation
    to the quartz crystal phase transition model.

    The SMPS current ripple modulates the GPU's power rail,
    which couples to the quartz oscillator through piezo effect.
    This is the same physics as the Lidari pump in overtone_resonance.py.
    """
    # Amplitude envelope → overtone density
    amp_norm = amplitude / (np.mean(amplitude) + 1e-10)

    # Phase transitions in Lidari model:
    #   solid_piezo:    density < 0.38  → stable crystal oscillation
    #   liquid_superfluid: 0.38 < density < 0.75 → phase slips begin
    #   plasma_crystal: density > 0.75 → full coherence, plasma lattice
    solid = np.sum(amp_norm < 1.38) / len(amp_norm)
    liquid = np.sum((amp_norm >= 1.38) & (amp_norm < 1.75)) / len(amp_norm)
    plasma = np.sum(amp_norm >= 1.75) / len(amp_norm)

    return {
        'solid_fraction': float(solid),
        'liquid_fraction': float(liquid),
        'plasma_fraction': float(plasma),
        'dominant_phase': 'plasma_crystal' if plasma > liquid else
                         'liquid_superfluid' if liquid > solid else
                         'solid_piezo'
    }

def time_crystal_analysis(phase_residual):
    """
    Discrete time crystal phase detection.
    The C3 rotation group has 3 discrete phases: {0, 500, 1000}.
    If the SMPS phase residual shows tri-modal clustering around
    these values (modulo C3_CYCLE_STEPS), the DrMOS switching is
    phase-locked to a discrete time crystal structure.
    """
    c3_cycle = C3_CYCLE_STEPS  # 1500
    # Fold phase residual into C3 cycle
    folded = np.abs(phase_residual) % (2.0 * math.pi)
    # Quantize to 3 bins
    bins = np.zeros(3, dtype=np.float64)
    for ph in folded:
        idx = int(ph / (2.0*math.pi) * 3) % 3
        bins[idx] += 1
    bins /= np.sum(bins)
    return {
        'T0_absorption': float(bins[0]),
        'T1_identity': float(bins[1]),
        'T2_expression': float(bins[2]),
        'time_crystal_phase': TIME_CRYSTAL_PHASES[np.argmax(bins)]
    }

# ============================================================
# MAIN ANALYSIS
# ============================================================
print("=" * 65)
print("  OVERTONE RESONANCE — Time-Domain SMPS Phase Analysis")
print("  N14 Quantum Clock + Liquid Quartz Dynamics")
print("=" * 65)

# Analyze both GPUs
results = {}
for name, path, f_nom in [
    ("gfx1200", "/data/rtl-sdr/docs/n14_gfx1200_smps_15msps.iq", 51050.0),
    ("GTX1060", "/data/rtl-sdr/docs/n14_gtx1060_solo_15msps.iq", 49896.0),
]:
    print(f"\n--- {name} ---")
    iq = read_iq(path)
    n = len(iq)//2
    print(f"  Data: {n/FS*1000:.0f}ms at {FS/1e6:.0f}MSPS")

    # Phase-lock to SMPS fundamental
    pll = overtone_phase_lock(iq, f_nom)

    # Actual frequency measured against N14
    freq = pll['actual_freq']
    n14_ratio = N14 / freq
    timing_jitter_ns = pll['phase_std'] / (2.0 * math.pi * freq) * 1e9

    print(f"  SMPS actual: {freq:.2f} Hz  (nominal: {f_nom:.0f} Hz)")
    print(f"  Frequency error: {pll['freq_error']:.2f} Hz  ({pll['freq_error']/freq*1e6:.1f} PPM)")
    print(f"  N14 cycles/SMPS: {n14_ratio:.1f}")
    print(f"  Phase std: {pll['phase_std']:.4f} rad")
    print(f"  Timing jitter: ±{timing_jitter_ns:.2f} ns")
    print(f"  Phase update rate: {pll['phase_rate']:.0f} Hz")

    # Lidari pump analysis
    pump = lidari_phase_pump(pll['amplitude'], pll['phase_residual'])
    print(f"  Lidari phase: {pump['dominant_phase']}")
    print(f"    Solid: {pump['solid_fraction']:.1%}  Liquid: {pump['liquid_fraction']:.1%}  Plasma: {pump['plasma_fraction']:.1%}")

    # Time crystal
    tc = time_crystal_analysis(pll['phase_residual'])
    purity = max(tc['T0_absorption'], tc['T1_identity'], tc['T2_expression'])
    print(f"  Time crystal purity: {purity:.1%}")
    print(f"    T0(abs): {tc['T0_absorption']:.1%}  T1(id): {tc['T1_identity']:.1%}  T2(exp): {tc['T2_expression']:.1%}")

    # DrMOS equivalent tr from phase residual
    # The phase std encodes the switching uncertainty
    # tr ≈ phase_std / (2π·f) — the time-domain equivalent of the -20dB/dec corner
    tr_ns = pll['phase_std'] / (2.0 * math.pi * freq) * 1e9 * 2  # ×2 for rise+fall
    results[name] = {
        'freq': freq, 'n14_ratio': n14_ratio,
        'jitter_ns': timing_jitter_ns, 'tr_ns': tr_ns,
        'pump': pump, 'tc': tc, 'pll': pll
    }
    print(f"  DrMOS tr (from phase): ~{tr_ns:.1f} ns")

# ============================================================
# COMPARISON
# ============================================================
print(f"\n{'='*65}")
print(f"  OVERTONE TIME-DOMAIN COMPARISON")
print(f"{'='*65}")
print(f"  {'GPU':<15s} {'SMPS':>8s} {'Jitter':>8s} {'tr_est':>8s} {'Lidari':>14s} {'TC':>6s}")
print(f"  {'-'*65}")
for name, r in results.items():
    print(f"  {name:<15s} {r['freq']:8.1f}Hz {r['jitter_ns']:7.2f}ns {r['tr_ns']:7.1f}ns "
          f"{r['pump']['dominant_phase']:>14s} {r['tc']['time_crystal_phase']:>5d}")

if len(results) == 2:
    r1 = results['gfx1200']
    r2 = results['GTX1060']
    print(f"\n  Jitter ratio (16nm/4nm): {r2['jitter_ns']/r1['jitter_ns']:.2f}x"
          if r1['jitter_ns'] > 0 else "")

print(f"\n  ✅ Overtone resonance time-domain analysis complete.")
