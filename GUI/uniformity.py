"""
Plot Saved Ray Tracing Data (Uniformity Metrics Version)
This script loads ray absorption data and plots heatmaps and histograms.

Uniformity metrics:
    - U  : Effective area uniformity index
    - CV : Coefficient of Variation

Both are computed on the top X% cumulative-energy region.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys


# ========================= CONFIGURATION =========================
DATA_PATH_DEFAULT = r"C:\Users\Zedd\OneDrive - Imperial College London\UROP\RESULTS\low_conc_ours"

HEATMAP_BINS_Y = 75
HEATMAP_BINS_Z = 100

Y_RANGE = (-3, 3)   # mm
Z_RANGE = (0, 8)    # mm

# *** KEY PARAMETER: cumulative energy percentage selection ***
ENERGY_THRESHOLD_PERCENT = 0.8   # <==== YOU CAN ADJUST THIS
# =================================================================



# ================================================================
# Select top cumulative-energy region
# ================================================================
def select_top_energy_region(counts, threshold):
    """
    Select bins containing the top 'threshold' fraction of total energy.
    Returns:
        mask : boolean mask same shape as counts (True = selected)
        cutoff_value : the minimum bin value included in the region
    """
    E = counts.flatten()
    total_energy = np.sum(E)

    # Sort from high to low
    E_sorted = np.sort(E)[::-1]
    cumulative = np.cumsum(E_sorted) / total_energy

    # Find how many bins needed to reach threshold
    k = np.searchsorted(cumulative, threshold)
    cutoff_value = E_sorted[k] if k < len(E_sorted) else 0

    # Produce mask (2D)
    mask = counts >= cutoff_value

    return mask, cutoff_value



# ================================================================
# Uniformity Metrics (computed on selected bins only)
# ================================================================
def compute_uniformity_metrics(counts, mask):
    """
    Compute U and CV only inside the selected mask region.
    """
    E = counts[mask].astype(float)
    N = len(E)

    if N == 0:
        return 0, 0

    total = np.sum(E)
    sum_sq = np.sum(E**2)

    U = (total**2) / (N * sum_sq)

    mean = total / N
    std = np.sqrt(np.mean((E - mean)**2))
    CV = std / mean if mean > 0 else 0

    return U, CV



# ================================================================
# Heatmap Plotting
# ================================================================
def plot_heatmap(ypos, zpos, energy_threshold):
    """
    Create 2D heatmap of absorption (Y on X-axis, Z on Y-axis).
    Computes U, CV on the top energy_threshold fraction of energy.
    Draws contour of selected region.
    """

    counts, y_edges, z_edges = np.histogram2d(
        ypos, zpos,
        bins=[HEATMAP_BINS_Y, HEATMAP_BINS_Z],
        range=[Y_RANGE, Z_RANGE]
    )

    # ---------------------- Select Top Energy Region ----------------------
    mask, cutoff_value = select_top_energy_region(counts, energy_threshold)
    U, CV = compute_uniformity_metrics(counts, mask)

    print("\n=== UNIFORMITY METRICS (Top {:.0f}% Energy) ===".format(energy_threshold * 100))
    print(f"Cutoff bin value      : {cutoff_value:.3f}")
    print(f"Selected bins         : {mask.sum()} / {counts.size}")
    print(f"U (effective-area)    : {U:.4f}")
    print(f"Wasted area (1 - U)   : {1 - U:.4f}")
    print(f"CV (variation)        : {CV:.4f}")

    # ------------------------------- Heatmap -------------------------------
    plt.figure(figsize=(7, 12), clear=True)
    plt.imshow(
        counts.T,
        origin='lower',
        extent=[y_edges[0], y_edges[-1], z_edges[0], z_edges[-1]],
        cmap='viridis',
        aspect='equal'
    )

    cbar = plt.colorbar(label='Absorbed rays')
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label('Absorbed rays', fontsize=16)

    plt.xlabel("Y position (mm)", fontsize=16)
    plt.ylabel("Z position (mm)", fontsize=16)
    plt.tick_params(axis='both', labelsize=14)

    # -------------------------------- Contour -------------------------------
    Y_centers = (y_edges[:-1] + y_edges[1:]) / 2
    Z_centers = (z_edges[:-1] + z_edges[1:]) / 2
    YY, ZZ = np.meshgrid(Y_centers, Z_centers)

    plt.contour(
        YY, ZZ, mask.T.astype(float),
        levels=[0.5],
        colors='red',
        linewidths=2,
        linestyles='solid'
    )

    plt.text(
        0.02, 0.98,
        f"Top {energy_threshold*100:.0f}% Energy Region\n"
        f"U = {U:.4f}\n"
        f"1 - U = {1 - U:.4f} (wasted)\n"
        f"CV = {CV:.4f}",
        transform=plt.gca().transAxes,
        verticalalignment='top',
        fontsize=14,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.85)
    )

    plt.title("Absorbed Energy Distribution (YZ Heatmap)", fontsize=18)
    plt.grid(False)
    plt.tight_layout()

    return counts, mask



# # ================================================================
# # 1D Histogram Plotting
# # ================================================================
# def plot_histogram(values, bins, x_range, title, xlabel, color):
#     plt.figure(clear=True)
#     plt.hist(values, bins=bins, range=x_range, alpha=0.7, color=color, edgecolor='black')
#     plt.title(title)
#     plt.xlabel(xlabel)
#     plt.ylabel("Absorbed rays")
#     plt.grid(True, alpha=0.3)

#     mean = np.mean(values)
#     std = np.std(values)

#     plt.text(
#         0.02, 0.98,
#         f"Mean = {mean:.2f}\nStd = {std:.2f}",
#         transform=plt.gca().transAxes,
#         verticalalignment='top',
#         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
#     )



# # ================================================================
# # Angle Histograms
# # ================================================================
# def plot_angle_histograms(dx, dy, dz):
#     az = np.degrees(np.arctan2(dy, dx))
#     az = (az + 360) % 360

#     plot_histogram(az, bins=36, x_range=(0, 360),
#                    title="Azimuthal Angle Distribution",
#                    xlabel="Azimuthal angle (deg)", color="purple")

#     cos_theta = np.abs(dz)
#     cos_theta = np.clip(cos_theta, 0, 1)
#     polar = np.degrees(np.arccos(cos_theta))

#     plot_histogram(polar, bins=90, x_range=(0, 90),
#                    title="Polar Angle Distribution",
#                    xlabel="Polar angle (deg)", color="orange")



# ================================================================
# Main Function
# ================================================================
def plot_saved_data(data_folder):

    data_path = Path(data_folder)
    csv_path = data_path / "absorbed_rays_raw_data.csv"

    if not csv_path.exists():
        print(f"Error: CSV file not found: {csv_path}")
        return

    print(f"\n=== LOADING DATA FROM: {csv_path} ===\n")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} absorbed rays")

    xpos = df["x_position"].values
    ypos = df["y_position"].values
    zpos = df["z_position"].values
    dx = df["direction_x"].values
    dy = df["direction_y"].values
    dz = df["direction_z"].values

    counts, mask = plot_heatmap(ypos, zpos, ENERGY_THRESHOLD_PERCENT)

    # NEW: Plot energy-area curve
    plot_energy_area_curve(counts, ENERGY_THRESHOLD_PERCENT)

    # # 1D histograms
    # plot_histogram(zpos, bins=50, x_range=Z_RANGE,
    #                title="Absorbed ray distribution along Z", xlabel="Z (mm)", color="blue")

    # plot_histogram(ypos, bins=50, x_range=Y_RANGE,
    #                title="Absorbed ray distribution along Y", xlabel="Y (mm)", color="green")

    # plot_histogram(xpos, bins=50, x_range=(-3, 3),
    #                title="Absorbed ray distribution along X", xlabel="X (mm)", color="red")

    # # Angle histograms
    # plot_angle_histograms(dx, dy, dz)

    print("\n=== ALL PLOTS GENERATED ===")
    plt.show()

# ================================================================
# Energy Fraction vs Area Fraction Curve
# ================================================================
def plot_energy_area_curve(counts, energy_threshold):
    """
    Plot the Energy Fraction vs Area Fraction curve,
    where both energy and area are restricted to the
    effective region that contains `energy_threshold`
    of the *total* absorbed energy.

    In this effective region:
        - x-axis: area fraction within the region (0 → 1)
        - y-axis: energy fraction within the region (0 → 1)
      For a perfectly uniform distribution in this region,
      the curve should be y = x.
    """

    # 1) 选出 top `energy_threshold` 能量对应的有效区域
    mask_eff, cutoff_eff = select_top_energy_region(counts, energy_threshold)
    E_eff = counts[mask_eff].astype(float)
    N_eff = E_eff.size

    if N_eff == 0 or np.sum(E_eff) == 0:
        print("Warning: No effective area found or zero energy in mask.")
        return

    # 方便打印：这块区域占全局能量的比例应该 ~ energy_threshold
    total_energy = np.sum(counts)
    eff_energy = np.sum(E_eff)
    eff_energy_fraction = eff_energy / total_energy if total_energy > 0 else 0

    # 2) 只在有效区域内排序、累计（完全丢掉指数尾巴区域）
    E_sorted = np.sort(E_eff)[::-1]  # 从大到小
    cum_energy_eff = np.cumsum(E_sorted) / np.sum(E_sorted)  # 0 → 1
    area_fraction = np.arange(1, N_eff + 1) / N_eff          # 0 → 1

    # 3) 作图
    plt.figure(figsize=(7, 6), clear=True)

    # 能量–面积曲线（有效域内）
    plt.plot(
        area_fraction,
        cum_energy_eff,
        linewidth=3,
        label="Energy–Area"
    )

    # 理想均匀分布：y = x
    plt.plot(
        [0, 1], [0, 1],
        'k--', linewidth=1.5, alpha=0.7,
        label="Perfect uniformity"
    )

    plt.xlabel("Area Fraction (within top {:.0f}% energy region)".format(energy_threshold*100),
               fontsize=12)
    plt.ylabel("Energy Fraction (within that region)", fontsize=12)
    plt.title("Energy Fraction vs Area Fraction (within top {:.0f}% energy region)".format(energy_threshold*100), fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend()

    plt.tight_layout()

    print(
        f"\nEnergy–Area curve (effective region only):\n"
        f"  Effective region contains ~{eff_energy_fraction*100:.1f}% of total energy "
        f"(target: {energy_threshold*100:.1f}%)\n"
        f"  Number of bins in effective region: {N_eff}"
    )

# ================================================================
# Entry Point
# ================================================================
if __name__ == "__main__":
    if len(sys.argv) > 1:
        folder = sys.argv[1]
    else:
        folder = DATA_PATH_DEFAULT
        print("\nUsing default path (edit in script if needed):")
        print(folder, "\n")

    plot_saved_data(folder)
