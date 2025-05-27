import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.animation import PillowWriter
# Plot matched-filtered signal before squint
def plot_matched_filtered_signal(rs, rc, c, dt, nn, v):
    # range-compressed signal across the synthetic aperture (v axis):
    # Aperture v — the cross-range path of the platform.
    # Fast-time t — approximates range.
    # Bright diagonal lines: Return signals from reflectors at different slant ranges that shift across the aperture due to motion → these are correct and expected pre-focusing effects
    tm = (2 * rc / c) + dt * (np.arange(nn) - nn / 2)
    plt.figure()
    plt.imshow(np.abs(rs), extent=[v[0], v[-1], tm[0], tm[-1]], aspect='auto', cmap='jet', origin='lower')
    plt.title("Matched-Filtered Signal (Before Squint Correction)")
    # v = slow-time (aperture traversal)
    plt.xlabel("Synthetic Aperture Position v (m)")
     # t = time within pulse = range delay
    plt.ylabel("Fast-Time (s) → Slant Range") 
    plt.colorbar(label='Amplitude')
    plt.savefig('plots/matched_filtered_signal.png', dpi=300)


# Plot range walk (single aperture column)
def plot_range_walk(rs, t, mm):
    # Peaks in this plot show the raw, uncorrected distance to targets.
    plt.figure()
    mid_aperture_idx = mm // 2
    plt.plot(t, np.abs(rs[:, mid_aperture_idx]))
    plt.title("Range Profile at Center of Synthetic Aperture")
    plt.xlabel("Fast-Time (s) → Slant Range")
    plt.ylabel("Echo Amplitude")
    plt.savefig('plots/range_walk_center_aperture.png', dpi=300)   


# Animate range profiles across aperture
# Shows how the range profile changes as radar aperture moves.
def animate_range_profiles(rs, t, v, mm):
    fig, ax = plt.subplots()
    line, = ax.plot([], [], lw=2)
    ax.set_xlim(t[0], t[-1])
    ax.set_ylim(0, np.max(rs))
    ax.set_xlabel("Fast-Time (s) → Slant Range")
    ax.set_ylabel("Echo Amplitude")
    ax.set_title("Range Profile Across Aperture Positions")

    def update(frame):
        line.set_data(t, rs[:, frame])
        ax.set_title(f'Aperture Index: {frame}  |  v = {v[frame]:.2f} m')
        return line,

    ani = FuncAnimation(fig, update, frames=range(mm), blit=True)
    ani.save("plots/range_profile_animation.gif", writer=PillowWriter(fps=15))

# 3D surface plot
def surface_plot_3d(rs, t, v):
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    T, V = np.meshgrid(t, v, indexing='ij')
    ax.plot_surface(V, T, rs, cmap='jet')
    ax.set_title("3D View: Range Profiles Over Synthetic Aperture")
    ax.set_xlabel("Aperture Position v (m)")
    ax.set_ylabel("Fast-Time t (s) → Slant Range")
    ax.set_zlabel("Echo Amplitude")
    fig.tight_layout()
    plt.savefig("plots/matched_filtered_surface.png", dpi=300)

# Plot actual target layout
def target_layout(px, uc, u0, v0, vc):
    plt.figure()
    plt.scatter(px[:, 0], px[:, 1], s=40, c='blue', edgecolors='k')
    plt.xlim([uc - u0, uc + u0])
    plt.ylim([vc - v0, vc + v0])
    plt.title("True Target Positions in (u,v) Plane")
    # Fast-time → Range resolution (per pulse)
    plt.xlabel("Range (u), meters")
    # Slow-time → Cross-range resolution (aperture synthesis)
    plt.ylabel("Cross-range (v), meters")
    plt.savefig('plots/targets_layout.png', dpi=300)

def drwchrt(rr, ux, vy, overlay_targets=None):
    plt.figure()
    ss = np.abs(rr).T
    xg, ng = np.max(ss), np.min(ss)
    cg = 255.0 / (xg - ng)
    image_data = 256 - cg * (ss - ng)
    plt.imshow(image_data, extent=(ux[0], ux[1], vy[0], vy[1]), cmap='gray', origin='lower', aspect='auto')
    plt.title('PROCESSED IMAGE OF TARGET AREA')
    plt.xlabel("Range (u) [m]")
    # Cross-range resolution (aperture synthesis)
    plt.ylabel("Cross-Range (v) [m]")
    if overlay_targets is not None:
        px = np.array(overlay_targets)
        plt.scatter(px[:, 0], px[:, 1], s=10, c='red', edgecolors='white', label='True Targets')
        plt.legend()
    plt.savefig('plots/sar_output.png', dpi=300)
