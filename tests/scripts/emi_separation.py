#!/usr/bin/env python3
"""EMI Separation: isolate pure VRM phase noise from broadband EMI.
Usage: python3 emi_separation.py <dpll_idle.csv> <dpll_load.csv> <f0_hz> <fs_dec_hz>
"""
import numpy as np, sys
from scipy.signal import welch

def main():
    idle_csv, load_csv = sys.argv[1], sys.argv[2]
    f0 = float(sys.argv[3]) if len(sys.argv)>3 else 51050.0
    fs_dec = float(sys.argv[4]) if len(sys.argv)>4 else 15000.0

    print(f"EMI Separation: {idle_csv} vs {load_csv}")
    print(f"  f0={f0} Hz  fs_dec={fs_dec} Hz\n")

    idle = np.loadtxt(idle_csv, delimiter=',', skiprows=1, max_rows=300000)
    load = np.loadtxt(load_csv, delimiter=',', skiprows=1, max_rows=300000)

    pe_i = idle[1000:,1]; pe_l = load[1000:,1]
    pe_i -= np.polyval(np.polyfit(np.arange(len(pe_i)), pe_i, 1), np.arange(len(pe_i)))
    pe_l -= np.polyval(np.polyfit(np.arange(len(pe_l)), pe_l, 1), np.arange(len(pe_l)))
    N = min(len(pe_i), len(pe_l))
    pe_i = pe_i[:N]; pe_l = pe_l[:N]

    nperseg = int(fs_dec)  # 1 Hz resolution
    f, Pi = welch(pe_i, fs=fs_dec, nperseg=nperseg, return_onesided=True)
    f, Pl = welch(pe_l, fs=fs_dec, nperseg=nperseg, return_onesided=True)

    # EMI scaling from 200-500 Hz
    mask_emi = (f >= 200) & (f <= 500)
    k_emi = np.mean(Pl[mask_emi] / (Pi[mask_emi] + 1e-30))
    print(f"EMI scale factor k = {k_emi:.2f}")

    # Clean VRM phase noise
    P_vrm = Pl - k_emi * Pi
    P_vrm[P_vrm < 0] = 0

    # Band analysis
    for flo, fhi, label in [(0.1,10,"0.1-10 Hz"),(1,5,"1-5 Hz"),(5,10,"5-10 Hz"),(0.3,0.7,"0.5 Hz")]:
        m = (f>=flo)&(f<=fhi)
        vi = np.sqrt(np.trapz(Pi[m], f[m]))/(2*np.pi*f0)*1e9
        vl = np.sqrt(np.trapz(Pl[m], f[m]))/(2*np.pi*f0)*1e9
        vc = np.sqrt(np.trapz(P_vrm[m], f[m]))/(2*np.pi*f0)*1e9
        print(f"  {label:>12s}:  idle={vi:.1f}ns  load_raw={vl:.1f}ns  load_clean={vc:.1f}ns  Δ={vc-vi:+.1f}ns ({(vc/vi-1)*100:+.1f}%)")

    print("\n✅ EMI separation complete.")

if __name__ == "__main__":
    main()
