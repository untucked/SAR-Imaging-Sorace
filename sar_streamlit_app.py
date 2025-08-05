import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from NoisySARFFTPolFmt import get_config, main as run_sar_sim
import os
import json
from streamlit_drawable_canvas import st_canvas
from PIL import Image, ImageDraw

st.set_page_config(layout="wide")
st.title("🎯 SAR Imaging via Chirp Scaling Algorithm")

# Sidebar inputs
st.sidebar.header("Input Parameters")
cfg = get_config()

# Inputs converted to MHz, MHz, us for UI (convert back to Hz or seconds later)
fc_mhz = st.sidebar.number_input("Center frequency (MHz)", value=cfg['fc'] / 1e6, format="%.2f")
df_mhz = st.sidebar.number_input("Half-bandwidth (MHz)", value=cfg['df'] / 1e6, format="%.2f")
tu_us = st.sidebar.number_input("Pulse duration (μs)", value=cfg['tu'] * 1e6, format="%.2f")

L = st.sidebar.number_input("Aperture length L (m)", value=float(cfg['L']))

uc = st.sidebar.number_input("Target region center uc (range, m)", value=float(cfg['uc']))
vc = st.sidebar.number_input("Target region center vc (cross-range, m)", value=float(cfg['vc']))

u0 = st.sidebar.number_input("Target region width u0 (range extent, m)", value=float(cfg['u0']))
v0 = st.sidebar.number_input("Target region width v0 (cross-range extent, m)", value=float(cfg['v0']))

u1 = st.sidebar.number_input("Initial u-position (m)", value=float(cfg['u1']))
v1 = st.sidebar.number_input("Initial v-position (m)", value=float(cfg['v1']))

ufc = st.sidebar.number_input("Oversampling factor ufc (fast-time)", value=int(cfg['ufc']))
vfc = st.sidebar.number_input("Oversampling factor vfc (slow-time)", value=int(cfg['vfc']))

snra = st.sidebar.number_input("Additive noise power (snra)", value=float(cfg['snra']))
snrm = st.sidebar.number_input("Multiplicative noise power (snrm)", value=float(cfg['snrm']))

# Convert UI units back to base units
fc = fc_mhz * 1e6
df = df_mhz * 1e6
tu = tu_us * 1e-6

# Store updates
cfg.update({
    'fc': fc, 'df': df, 'tu': tu,
    'L': L, 'uc': uc, 'vc': vc,
    'u0': u0, 'v0': v0,
    'u1': u1, 'v1': v1,
    'ufc': ufc, 'vfc': vfc,
    'snra': snra, 'snrm': snrm
})

st.subheader("🎯 Define Target Locations (Click to Add/Delete)")

canvas_size = 400

# Create a heatmap grid background
grid_np = 255 * np.ones((canvas_size, canvas_size, 3), dtype=np.uint8)

# Add grid lines
for x in range(0, canvas_size, 40):
    grid_np[:, x:x+1] = [200, 200, 200]  # vertical lines
for y in range(0, canvas_size, 40):
    grid_np[y:y+1, :] = [200, 200, 200]  # horizontal lines

background_image = Image.fromarray(grid_np)

if 'targets' not in st.session_state:
    st.session_state.targets = []
if 'reflectivities' not in st.session_state:
    st.session_state.reflectivities = []

canvas_result = st_canvas(
    fill_color="rgba(255, 0, 0, 0.6)",
    stroke_width=1,
    stroke_color="#ff0000",
    background_image=background_image,
    height=canvas_size,
    width=canvas_size,
    drawing_mode="point",
    key="canvas"
)

# Convert canvas points to (u, v) coordinates
if canvas_result.json_data is not None:
    objects = canvas_result.json_data["objects"]
    new_targets = []
    for obj in objects:
        cx = obj["left"]
        cy = obj["top"]
        v_range = vc - v0, vc + v0
        u_range = uc - u0, uc + u0
        v = v_range[0] + (cx / canvas_size) * (v_range[1] - v_range[0])
        u = u_range[0] + (cy / canvas_size) * (u_range[1] - u_range[0])
        new_targets.append([u, v])

    # Create a simple key for each target based on rounded coords
    def target_key(t):
        return f"{round(t[0],1)}_{round(t[1],1)}"
    existing_keys = {f"{round(t[0],1)}_{round(t[1],1)}" for t in st.session_state.targets}
    # Save newly clicked points
    for t in new_targets:
        k = target_key(t) 
        if k not in existing_keys:
            st.session_state.targets.append(t)
            st.session_state.reflectivities.append(1.0)  # default reflectivity

    # Update reflectivity inputs
    updated_reflectivities = []
    for i, t in enumerate(st.session_state.targets):
        refl = st.number_input(
            f"Reflectivity for target at ({t[0]:.1f}, {t[1]:.1f})",
            min_value=0.0, value=st.session_state.reflectivities[i],
            key=f"refl_input_{i}"
        )
        updated_reflectivities.append(refl)

    st.session_state.reflectivities = updated_reflectivities


if st.button("Clear Targets"):
    st.session_state.targets = []
    st.session_state.reflectivities = []

st.write("Current Targets (u, v) with Reflectivity:")
table_data = [
    {"u (range)": f"{t[0]:.2f}", "v (cross-range)": f"{t[1]:.2f}", "Reflectivity": r}
    for t, r in zip(st.session_state.targets, st.session_state.reflectivities)
]
st.dataframe(table_data)

# Run simulation button
if st.button("🚀 Generate SAR Imaging"):
    cfg['px'] = st.session_state.targets
    cfg['fx'] = st.session_state.reflectivities

    with open("config.json", "w") as f:
        json.dump(cfg, f, indent=4)

    st.write("Running SAR imaging...")
    run_sar_sim()
    st.success("SAR imaging complete!")

    # Display outputs only if files exist
    def safe_show_image(path, caption):
        if os.path.exists(path):
            if path.endswith(".gif"):
                st.image(path, caption=caption, use_column_width=True)  # GIFs must use full width
            else:
                st.image(path, caption=caption, width=500)  # Still image → custom width
        else:
            st.warning(f"Image not found: {path}")
    st.subheader("📡 Output SAR Image")
    safe_show_image("plots/sar_output.png", "Final SAR Image")

    st.subheader("Target Layout Preview")
    safe_show_image("plots/targets_layout.png", "Target Layout")

    st.subheader("🧩 Additional Plots")
    safe_show_image("plots/matched_filtered_signal.png", "Matched Filtered Signal")
    safe_show_image("plots/range_walk_center_aperture.png", "Range Walk")
    safe_show_image("plots/matched_filtered_surface.png", "3D SAR Surface Plot")
    safe_show_image("plots/range_profile_animation.gif", "Animated Range Profiles")
