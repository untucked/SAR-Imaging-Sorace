# 🎯 SAR Imaging – Python FFT-Based Simulation  
![License](https://img.shields.io/badge/license-MIT-blue.svg)  
![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)  
![Numpy](https://img.shields.io/badge/numpy-1.24%2B-blue.svg)  
![Matplotlib](https://img.shields.io/badge/matplotlib-3.7%2B-blue.svg)

## 📡 Introduction
This is a Python port of a MATLAB-based **Noisy SAR FFT Polar Format Algorithm**. It simulates radar returns from multiple point targets, applies matched filtering, corrects for squint and range walk, and produces a **focused SAR image**.

It includes:
- Target modeling
- Noise injection
- Range compression
- Squint correction
- Full polar formatting and interpolation
- Visualization and animation tools

---

## 🎯 Goal
- Simulate airborne SAR imaging of sparse point targets
- Visualize signal evolution before and after matched filtering
- Demonstrate cross-range motion and synthetic aperture principles
- Generate plots, GIF animations, and final image output

---

## ⚙️ How It Works
1. **Target Modeling**: Hardcoded or JSON-configured target positions and strengths.
2. **Signal Simulation**: Models returns using chirp pulses, with optional Gaussian and Rayleigh noise.
3. **Matched Filtering**: Applies reference chirp and performs pulse compression.
4. **Squint Correction**: Corrects Doppler effects due to platform motion.
5. **FFT Processing**: Applies 2D FFT + interpolation to form the SAR image.
6. **Visualization**:
   - Target layout
   - Matched-filter signal vs fast-time
   - Aperture sweep animation (GIF)
   - Final SAR image with overlays

---

## ✨ Features
- ✅ Supports both noise-free and noisy SAR simulation
- 📈 Visual output at every key stage
- 🧭 Range walk visualization & squint correction
- 🎞️ Animated GIF of range profile sweep
- 🖼️ 3D surface plot of matched-filtered matrix
- 🗃️ Configurable via `config.json` (or falls back to `input()`)

---

## 📁 Project Structure
```
SAR_Imaging/
├── NoisySARFFTPolFmt.py       # Main script
├── support.py                 # FFT, interpolation, helpers
├── plotting.py                # Modular plotting logic
├── config.json                # Optional user-defined scenario
├── plots/                     # Output images and GIFs
└── README.md                  # This file
```

---

## 🧩 Dependencies

Install with:
```bash
pip install numpy matplotlib pandas
```

Additional (optional for animations):
- `ffmpeg` (for `.mp4` animations) or
- `Pillow` (for `.gif`)

---

## 🧪 Run It

```bash
python NoisySARFFTPolFmt.py
```
---

## 🖼️ Sample Output

- ✅ `plots/targets_layout.png`: Ground truth target positions
- ✅ `plots/matched_filtered_signal.png`: Compressed signal vs aperture
- ✅ `plots/range_profile_animation.gif`: Range walk over time
- ✅ `plots/matched_filtered_surface.png`: 3D matched filter matrix
- ✅ `plots/sar_output.png`: Final SAR image

---

## GIT - CLONE

``` bash
git clone https://github.com/untucked/SAR-Imaging-Sorace.git
```

## 🔧 Git Setup

```bash
git init
git add .
git commit -m "Initial SAR Imaging Commit"
git branch -M main
git remote add origin https://github.com/untucked/SAR-Imaging-Sorace.git
git push -u origin main
```

