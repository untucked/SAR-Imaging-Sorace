import numpy as np
import matplotlib.pyplot as plt

plt.style.use('ggplot')

def dftx(s):
    return np.fft.fftshift(np.fft.fft(np.fft.fftshift(s, axes=0), axis=0), axes=0)

def dfty(s):
    return np.fft.fftshift(np.fft.fft(np.fft.fftshift(s.T, axes=0), axis=0), axes=0).T

def idftx(fs):
    return np.fft.ifftshift(np.fft.ifft(np.fft.ifftshift(fs, axes=0), axis=0), axes=0)

def idfty(fs):
    return np.fft.ifftshift(np.fft.ifft(np.fft.ifftshift(fs.T, axes=0), axis=0), axes=0).T

def intrplt(xp, x, y, npt, dx):
    pt = np.zeros_like(xp, dtype=complex)
    slb = 8
    for j, xpt in enumerate(xp):
        ds = np.abs(xpt - x[0])
        ic = 0
        for i in range(npt - 1):
            dn = np.abs(xpt - x[i])
            if dn < ds:
                ic = i
                ds = dn
        if ds == 0:
            pt[j] = y[ic]
        else:
            l1 = max(ic - slb, 0)
            l2 = min(ic + slb, npt - 1)
            for i in range(l1, l2 + 1):
                xa = (x[i] - xpt) / dx
                snc = np.sinc(xa / np.pi) if xa != 0 else 1.0
                hm = 0.54 + 0.46 * np.cos(np.pi * xa / slb)
                pt[j] += y[i] * snc * hm
    return pt

