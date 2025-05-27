import numpy as np
import matplotlib.pyplot as plt
import json
import os
import pandas as pd
from numpy import pi, sqrt, exp, sin, cos
import sys

# local
from support import dftx, dfty, idftx, idfty, intrplt
from plotting import plot_matched_filtered_signal, plot_range_walk, animate_range_profiles, surface_plot_3d, target_layout, drwchrt
# import ggplot
plt.style.use('ggplot')

# ========================
# Load config or prompt user
# ========================
def get_config():
    config_path = 'config.json'
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            cfg = json.load(f)
    else:
        print(f"'{config_path}' not found. Falling back to default configuration...")
        cfg = {
            "fc": 2e8,
            "df": 5e7,
            "tu": 2.5e-7,
            "L": 100,
            "uc": 1000,
            "vc": 300,
            "u0": 20,
            "v0": 60,
            "u1": 0,
            "v1": 0,
            "ufc": 2,
            "vfc": 2,
            "snra": 0,
            "snrm": 0,
            # It's good practice to convert these to numpy arrays if they'll be used as such later
            "px": np.array([[1000, 300], [1014, 264], [1000, 249],
                            [990, 345], [992, 277], [1012, 340], [1005, 325]]),
            "fx": np.array([1.0, 2.4, 0.5, 1.0, 2.2, 2.0, 2.0])
        }
    return cfg


# ========================
# Main SAR simulation
# ========================

def main():
    os.makedirs('plots', exist_ok=True)
    cfg = get_config()
    fc   = cfg['fc']   # Center frequency of the chirped pulse (Hz)
    df   = cfg['df']   # Half-bandwidth of the pulse (Hz), so total bandwidth = 2*df
    tu   = cfg['tu']   # Pulse duration (seconds), also determines chirp slope

    L    = cfg['L']    # Physical aperture length (m) — how far the radar platform moves

    uc   = cfg['uc']   # Center of the target region in range (u-direction)
    vc   = cfg['vc']   # Center of the target region in cross-range (v-direction)

    u0   = cfg['u0']   # Width of the target region in u (range extent)
    v0   = cfg['v0']   # Width of the target region in v (cross-range extent)

    u1   = cfg['u1']   # Initial u-position of the radar platform (typically 0)
    v1   = cfg['v1']   # Initial v-position of the radar platform (along aperture)

    ufc  = cfg['ufc']  # Oversampling factor for fast-time (pulse sampling)
    vfc  = cfg['vfc']  # Oversampling factor for slow-time (aperture sampling)

    snra = cfg['snra'] # Additive noise power (SNR as a power ratio, e.g., 0 = no noise)
    snrm = cfg['snrm'] # Multiplicative noise power (e.g., Rayleigh scaling)

    px   = np.array(cfg['px'])  # Target positions, shape (n_targets, 2), columns: [u, v]
    fx   = np.array(cfg['fx'])  # Target reflectivity (amplitude response), shape (n_targets,)
    ntr  = len(fx)              # Number of targets


    # Example parameters
    c = 3e8
    pi2 = 2 * np.pi
    ci = 1j

    # Derived constants
    fl = fc - df
    fh = fc + df
    fp = 4 * ufc * df
    ap = df / tu
    dt = 1 / fp
    thta = np.arctan((vc - v1) / uc)
    drm = sqrt(u0**2 + v0**2)
    tm = 1 / fh
    kc = pi2 * fc / c

    # Pulse repetition
    dv = uc * c / (4 * v0 * (cos(thta)**2))
    dm = dv * tm / (1.2 * vfc)
    mm = int(2 * np.ceil(L / dm))
    dkv = pi / L

    # Pulse sampling
    d1 = sqrt((uc - u0)**2 + (vc - v0 - v1 - L)**2)
    d2 = sqrt((uc + u0)**2 + (vc + v0)**2)
    rmin = d1 if (vc - v0 > v1 + L) else d2
    Ts = 2 * rmin / c
    rmax = sqrt((uc + u0)**2 + (vc + v0 - v1 + L)**2)
    Te = 2 * rmax / c + tu
    tdd = Te - Ts
    Ts -= tdd / 10
    Te += tdd / 10
    dn = max((Te - Ts) / 2, drm / c)
    nn = int(2 * np.ceil(dn / dt))
    dk = pi / (c * dn)
    kd = 2 * kc * sin(thta)
    dkx = pi2 / max(c * dn, u0)

    ntr = len(fx)

    # Signal simulation
    rc = sqrt(uc**2 + vc**2)
    t = Ts + dt * np.arange(nnn := nn)
    v = v1 + dm * (np.arange(mm) - mm/2)
    rr = np.zeros((nnn, mm), dtype=complex)

    for k in range(ntr):
        rg = 2 * np.sqrt(px[k,0]**2 + (px[k,1] - v)**2) / c
        tt = np.outer(t, np.ones(mm)) - rg[np.newaxis, :]
        tx = pi2 * (fl + ap * tt) * tt
        t0 = (tt >= 0) & (tt <= tu)
        rr += fx[k] * np.exp(ci * tx) * t0

    if snra > 0:
        rr += sqrt(snra) * (np.random.normal(0, 1, rr.shape) + 1j * np.random.normal(0, 1, rr.shape))
    if snrm > 0:
        rr *= 1 + sqrt(snrm) * np.random.rayleigh(scale=1.0, size=rr.shape)
    
    # Plot actual target layout
    target_layout(px, uc, u0, v0, vc)

    # Reference signal
    tt = np.outer(t, np.ones(mm)) - (2 * rc / c)
    t2 = (tt >= 0) & (tt <= tu)
    tx = pi2 * (fl + ap * tt) * tt
    sr = np.exp(ci * tx) * t2

    # Downconvert
    tx = pi2 * fc * np.outer(t, np.ones(mm))
    cn = np.exp(-ci * tx)
    rr *= cn
    sr *= cn

    # FFT
    rr = dftx(rr)
    sr = dftx(sr)
    rr *= np.conj(sr)
    rs = idftx(rr)
    # Plot matched-filtered signal before squint
    plot_matched_filtered_signal(rs, rc, c, dt, nn, v)
    # Plot range walk (single aperture column)
    plot_range_walk(rs, t, mm)
    # Animate range profiles across aperture
    animate_range_profiles(rs, t, v, mm)
    # 3D surface plot
    surface_plot_3d(rs, t, v)
    # Peak amplitude and estimated range for center aperture
    mid_col = rs[:, mm // 2]
    peak_idx = np.argmax(mid_col)
    peak_time = t[peak_idx]
    estimated_range = 3e8 * peak_time / 2  # simple c*t/2 formula

    peak_df = pd.DataFrame({
        "Peak Amplitude": [mid_col[peak_idx]],
        "Peak Time (s)": [peak_time],
        "Estimated Range (m)": [estimated_range]
    })
    print(peak_df.to_string(index=False))
    peak_df.to_csv('plots/peak_estimation.csv', index=False)

    # Squint correction
    kf = kd * dm * (np.arange(mm) - mm/2)
    cn = np.exp(-ci * np.outer(np.ones(nn), kf))
    rr *= cn

    # Spatial transform
    rr = dfty(rr)

    # Match filtering in frequency space
    kv = kd + dkv * (np.arange(mm) - mm / 2)
    kr = kc + dk * (np.arange(nnn) - nnn / 2)
    kvv = kv**2
    krr = kr**2

    # MATLAB: ks = 4 * krr(:) * ones(1,mm) - ones(nn,1) * kvv;
    # ks = 4 * np.outer(krr, np.ones(mm)) - np.outer(np.ones(nnn), kvv)
    # ks = np.maximum(ks, 0)
    ks = 4 * krr.reshape(-1, 1) - kvv.reshape(1, -1)
    ks = ks * (ks > 0) # Element-wise multiplication for boolean masking
    ku = np.sqrt(ks)

    # Phase factor for filter
    # MATLAB: tx = ku * uc + ones(nn,1) * kv * vc;
    tx = ku * uc + np.ones((nn, 1)) * (kv * vc)
    # MATLAB: tx = tx - 2 * (kr(:) * ones(1,mm)) * rc;
    tx = tx - 2 * (kr.reshape(-1, 1) * np.ones((1, mm))) * rc
    cn = np.exp(ci * tx)  # Filter signal
    rr *= cn

    # Interpolation
    kx = kc + dkx * (np.arange(nn) - nn/2)
    for i in range(mm):
        rr[:, i] = intrplt(kx, ku[:, i], rr[:, i], nn, dkx)

    # Inverse FFTs
    rr = idfty(rr)
    rr = idftx(rr)

    # Final plot
    # Range increment
    dlu = max(np.pi * np.cos(thta) / (2 * dk), u0)
    # Cross range increment
    dlv = max(4 * L * v0 * np.cos(thta) / uc, L)
    # Range limits
    u_x = [uc - dlu, uc + dlu]
    # Cross range limits
    v_y = [vc - dlv, vc + dlv]
    drwchrt(rr, u_x, v_y, overlay_targets=px)

if __name__ == '__main__':
    main()
    # plt.show()
