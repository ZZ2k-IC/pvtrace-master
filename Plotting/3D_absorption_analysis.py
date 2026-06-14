"""
Plot Saved Ray Tracing Data (3D Uniformity Metrics Version)
This script loads ray absorption data and plots 2D projections and histograms.

Uniformity metrics:
    - U  : Effective volume uniformity index (computed in 3D)
    - CV : Coefficient of Variation (computed in 3D)

Both are computed on the top X% cumulative-energy region in 3D.
Visualization (heatmaps, gradients) are 2D projections of the 3D distribution.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import plotly.graph_objects as go


# ========================= CONFIGURATION =========================
DATA_PATH_DEFAULT = r"D:Results/CeYAG"  # Primary dataset folder path

# 3D histogram bins
HEATMAP_BINS_X = 60
HEATMAP_BINS_Y = 60
HEATMAP_BINS_Z = 80

# Spatial ranges (mm)
X_RANGE = (-3, 3)   # you can adjust
Y_RANGE = (-3, 3)       # mm
Z_RANGE = (0, 8)        # mm

# *** KEY PARAMETER: cumulative energy percentage selection in 3D ***
ENERGY_THRESHOLD_PERCENT = 0.8  
# =================================================================

# Excel export output path
EXCEL_OUTPUT_PATH = None  # <==== YOU CAN ADJUST THIS PATH


# ================================================================
# Angular Distribution Plotting
# ================================================================
def plot_angular_distributions(dx, dy, dz, title_prefix=""):
    """
    Plot azimuthal and polar angle distributions.
    
    Args:
        dx, dy, dz: Direction components
        title_prefix: Prefix for plot titles (e.g., "Absorbed" or "Detected")
    """
    # Azimuthal angle distribution
    plt.figure(figsize=(10, 4), clear=True)
    
    plt.subplot(1, 2, 1)
    az = np.degrees(np.arctan2(dy, dx))
    az = (az + 360) % 360
    plt.hist(az, bins=36, range=(0, 360), alpha=0.7, color="purple", edgecolor='black')
    plt.title(f"{title_prefix} Azimuthal Angle Distribution", fontsize=14)
    plt.xlabel("Azimuthal angle (deg)", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.grid(True, alpha=0.3)
    
    mean_az = np.mean(az)
    std_az = np.std(az)
    plt.text(
        0.02, 0.98,
        f"Mean = {mean_az:.2f}°\nStd = {std_az:.2f}°",
        transform=plt.gca().transAxes,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
        fontsize=10
    )
    
    # Polar angle distribution
    plt.subplot(1, 2, 2)
    cos_theta = np.abs(dz)
    cos_theta = np.clip(cos_theta, 0, 1)
    polar = np.degrees(np.arccos(cos_theta))
    plt.hist(polar, bins=90, range=(0, 90), alpha=0.7, color="orange", edgecolor='black')
    plt.title(f"{title_prefix} Polar Angle Distribution", fontsize=14)
    plt.xlabel("Polar angle (deg)", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.grid(True, alpha=0.3)
    
    mean_polar = np.mean(polar)
    std_polar = np.std(polar)
    plt.text(
        0.02, 0.98,
        f"Mean = {mean_polar:.2f}°\nStd = {std_polar:.2f}°",
        transform=plt.gca().transAxes,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
        fontsize=10
    )
    
    plt.tight_layout()
    print(f"\n{title_prefix} angular distributions plotted (azimuthal and polar angles).")


# ================================================================
# 3D Histogram
# ================================================================
def make_3d_histogram(xpos, ypos, zpos):
    """
    Build a 3D histogram of absorbed rays over (x, y, z).
    Returns:
        counts3d : 3D array (Nx, Ny, Nz)
        x_edges, y_edges, z_edges : bin edges for each axis
    """
    counts3d, edges = np.histogramdd(
        np.vstack((xpos, ypos, zpos)).T,
        bins=[HEATMAP_BINS_X, HEATMAP_BINS_Y, HEATMAP_BINS_Z],
        range=[X_RANGE, Y_RANGE, Z_RANGE]
    )
    x_edges, y_edges, z_edges = edges
    return counts3d, x_edges, y_edges, z_edges


# ================================================================
# Select top cumulative-energy region (works for any dimension)
# ================================================================
def select_top_energy_region(counts, threshold):
    """
    Select bins containing the top 'threshold' fraction of total energy.
    counts can be 2D or 3D; internally flattened.
    
    Returns:
        mask : boolean mask same shape as counts (True = selected)
        cutoff_value : the minimum bin value included in the region
    """
    E = counts.flatten().astype(float)
    total_energy = np.sum(E)
    if total_energy <= 0:
        return np.zeros_like(counts, dtype=bool), 0.0

    # Sort from high to low
    E_sorted = np.sort(E)[::-1]
    cumulative = np.cumsum(E_sorted) / total_energy

    # Find how many bins needed to reach threshold
    k = np.searchsorted(cumulative, threshold)
    cutoff_value = E_sorted[k] if k < len(E_sorted) else 0

    # Produce mask (same shape as counts)
    mask = counts >= cutoff_value

    return mask, cutoff_value


# ================================================================
# Uniformity Metrics (computed on selected bins only)
# ================================================================
def compute_uniformity_metrics(counts, mask):
    """
    Compute U and CV only inside the selected mask region.
    counts can be 2D or 3D; mask must be same shape.
    """
    E = counts[mask].astype(float)
    N = len(E)

    if N == 0:
        return 0.0, 0.0

    total = np.sum(E)
    if total <= 0:
        return 0.0, 0.0

    sum_sq = np.sum(E**2)
    U = (total**2) / (N * sum_sq) if sum_sq > 0 else 0.0

    mean = total / N
    std = np.sqrt(np.mean((E - mean)**2))
    CV = std / mean if mean > 0 else 0.0

    return U, CV


# ================================================================
# 2D Heatmap Projection (from 3D)
# ================================================================
def plot_heatmap_3d_projection(counts3d, x_edges, y_edges, z_edges, energy_threshold):
    """
    Create 2D YZ and XZ heatmaps by integrating the 3D distribution over X and Y respectively.
    Uniformity metrics (U, CV) are computed in full 3D on the top
    energy_threshold fraction of total energy.

    The red contour shows the projection of the 3D effective region onto each plane.
    """

    # ---------- 3D top-energy region & uniformity ----------
    mask3d, cutoff_value = select_top_energy_region(counts3d, energy_threshold)
    U3d, CV3d = compute_uniformity_metrics(counts3d, mask3d)

    print("\n=== 3D UNIFORMITY METRICS (Top {:.0f}% Energy) ===".format(energy_threshold * 100))
    print(f"Cutoff bin value      : {cutoff_value:.3f}")
    print(f"Selected voxels       : {mask3d.sum()} / {counts3d.size}")
    print(f"U_3D (effective-vol)  : {U3d:.4f}")
    print(f"Wasted volume (1-U_3D): {1 - U3d:.4f}")
    print(f"CV_3D (variation)     : {CV3d:.4f}")

    # ---------- YZ projection (sum over X) ----------
    counts_yz = counts3d.sum(axis=0)        # shape (Ny, Nz)
    mask_yz = mask3d.any(axis=0)           # projection of 3D effective region

    plt.figure(figsize=(7, 12), clear=True)
    plt.imshow(
        counts_yz.T,
        origin='lower',
        extent=[y_edges[0], y_edges[-1], z_edges[0], z_edges[-1]],
        cmap='viridis',
        aspect='equal',
        vmin = 0,
    )

    cbar = plt.colorbar(label='Absorbed rays (integrated over X)')
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label('Absorbed rays (∑ over X)', fontsize=16)

    plt.xlabel("Y position (mm)", fontsize=16)
    plt.ylabel("Z position (mm)", fontsize=16)
    plt.tick_params(axis='both', labelsize=14)

    # Contour: projection of 3D effective region onto YZ
    Y_centers = (y_edges[:-1] + y_edges[1:]) / 2
    Z_centers = (z_edges[:-1] + z_edges[1:]) / 2
    YY, ZZ = np.meshgrid(Y_centers, Z_centers)

    # plt.contour(
    #     YY, ZZ, mask_yz.T.astype(float),
    #     levels=[0.5],
    #     colors='red',
    #     linewidths=2,
    #     linestyles='solid'
    # )

    plt.text(
        0.02, 0.98,
        f"Top {energy_threshold*100:.0f}% Energy Region (3D)\n"
        f"U_3D = {U3d:.4f}\n"
        f"1 - U_3D = {1 - U3d:.4f}\n"
        f"CV_3D = {CV3d:.4f}",
        transform=plt.gca().transAxes,
        verticalalignment='top',
        fontsize=14,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.85)
    )

    plt.title("Absorbed Energy Projection onto YZ (∑ over X)", fontsize=18)
    plt.grid(False)
    plt.tight_layout()

    # ---------- XZ projection (sum over Y) ----------
    counts_xz = counts3d.sum(axis=1)        # shape (Nx, Nz)
    mask_xz = mask3d.any(axis=1)           # projection of 3D effective region

    plt.figure(figsize=(7, 12), clear=True)
    plt.imshow(
        counts_xz.T,
        origin='lower',
        extent=[x_edges[0], x_edges[-1], z_edges[0], z_edges[-1]],
        cmap='viridis',
        aspect='equal',
        vmin = 0,
    )

    cbar = plt.colorbar(label='Absorbed rays (integrated over Y)')
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label('Absorbed rays (∑ over Y)', fontsize=16)

    plt.xlabel("X position (mm)", fontsize=16)
    plt.ylabel("Z position (mm)", fontsize=16)
    plt.tick_params(axis='both', labelsize=14)

    # Contour: projection of 3D effective region onto XZ
    X_centers = (x_edges[:-1] + x_edges[1:]) / 2
    XX, ZZ_xz = np.meshgrid(X_centers, Z_centers)

    # plt.contour(
    #     XX, ZZ_xz, mask_xz.T.astype(float),
    #     levels=[0.5],
    #     colors='red',
    #     linewidths=2,
    #     linestyles='solid'
    # )

    plt.text(
        0.02, 0.98,
        f"Top {energy_threshold*100:.0f}% Energy Region (3D)\n"
        f"U_3D = {U3d:.4f}\n"
        f"1 - U_3D = {1 - U3d:.4f}\n"
        f"CV_3D = {CV3d:.4f}",
        transform=plt.gca().transAxes,
        verticalalignment='top',
        fontsize=14,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.85)
    )

    plt.title("Absorbed Energy Projection onto XZ (∑ over Y)", fontsize=18)
    plt.grid(False)
    plt.tight_layout()

    return mask3d, U3d, CV3d


# ================================================================
# Energy Fraction vs Area Fraction Curve (3D effective region)
# ================================================================
def plot_energy_area_curve(counts, energy_threshold, label="Energy–Area"):
    """
    Plot the Energy Fraction vs Area Fraction curve, restricted to the
    3D effective region that contains `energy_threshold` of the *total*
    absorbed energy.

    In this effective region:
        - x-axis: voxel fraction within the region (0 → 1)
        - y-axis: energy fraction within the region (0 → 1)
      For a perfectly uniform distribution in this region,
      the curve should be y = x.
    """

    # 1) Select top energy_threshold of total energy in 3D
    mask_eff, cutoff_eff = select_top_energy_region(counts, energy_threshold)
    E_eff = counts[mask_eff].astype(float)
    N_eff = E_eff.size

    if N_eff == 0 or np.sum(E_eff) == 0:
        print(f"Warning: No effective 3D region found for {label}.")
        return None, None

    total_energy = np.sum(counts)
    eff_energy = np.sum(E_eff)
    eff_energy_fraction = eff_energy / total_energy if total_energy > 0 else 0.0

    # 2) Sort energies within effective region
    E_sorted = np.sort(E_eff)[::-1]  # high -> low
    cum_energy_eff = np.cumsum(E_sorted) / np.sum(E_sorted)  # 0 → 1
    voxel_fraction = np.arange(1, N_eff + 1) / N_eff         # 0 → 1

    print(
        f"\n{label} (3D effective region only):\n"
        f"  Effective region contains ~{eff_energy_fraction*100:.1f}% of total energy "
        f"(target: {energy_threshold*100:.1f}%)\n"
        f"  Number of voxels in effective region: {N_eff}"
    )

    # 3) Plot
    plt.figure(figsize=(7, 6), clear=True)

    plt.plot(
        voxel_fraction,
        cum_energy_eff,
        linewidth=3,
        label=label
    )

    # Perfect uniformity
    plt.plot(
        [0, 1], [0, 1],
        'k--', linewidth=1.5, alpha=0.7,
        label="Perfect uniformity"
    )

    plt.xlabel("Voxel Fraction (within top {:.0f}% energy region)".format(energy_threshold*100),
               fontsize=12)
    plt.ylabel("Energy Fraction (within that region)", fontsize=12)
    plt.title("Energy Fraction vs Voxel Fraction (3D effective region)", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    
    # Save data to Excel
    try:
        output_path = Path(EXCEL_OUTPUT_PATH)
        output_path.mkdir(parents=True, exist_ok=True)
        
        df_export = pd.DataFrame({
            'Voxel_Fraction': voxel_fraction,
            'Energy_Fraction': cum_energy_eff
        })
        excel_filename = output_path / f"energy_voxel_curve_{label.replace(' ', '_').replace('–', '-')}.xlsx"
        df_export.to_excel(excel_filename, index=False)
        print(f"\n✓ Data saved to: {excel_filename}")
    except ImportError:
        print("\n✗ ERROR: openpyxl library not installed. Install it with: pip install openpyxl")
    except Exception as e:
        print(f"\n✗ ERROR saving Excel file: {e}")
    
    return voxel_fraction, cum_energy_eff



# ================================================================
# Gradient Map (from 3D -> 2D projection)
# ================================================================
def plot_gradient_map(counts3d, x_edges, y_edges, z_edges):
    """
    Compute 3D gradient of E(x,y,z), then project the gradient magnitude
    onto the YZ plane (e.g. average over X) to visualize local non-uniformity.
    """

    # 3D gradients
    dEx, dEy, dEz = np.gradient(counts3d)
    grad_mag3d = np.sqrt(dEx**2 + dEy**2 + dEz**2)

    # Project onto YZ (e.g. average over X)
    grad_yz = grad_mag3d.mean(axis=0)  # shape (Ny, Nz)

    Y_centers = (y_edges[:-1] + y_edges[1:]) / 2
    Z_centers = (z_edges[:-1] + z_edges[1:]) / 2
    YY, ZZ = np.meshgrid(Y_centers, Z_centers)

    plt.figure(figsize=(7, 12), clear=True)
    plt.imshow(
        grad_yz.T,
        origin='lower',
        extent=[y_edges[0], y_edges[-1], z_edges[0], z_edges[-1]],
        cmap="inferno",
        aspect='equal',
    )

    cbar = plt.colorbar(label='|∇E| (Gradient Magnitude, projected)')
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label('|∇E| (Gradient Magnitude)', fontsize=16)

    plt.title("Gradient Map (3D → YZ projection)", fontsize=18)
    plt.xlabel("Y position (mm)", fontsize=16)
    plt.ylabel("Z position (mm)", fontsize=16)
    plt.tick_params(axis='both', labelsize=14)
    plt.grid(False)
    plt.tight_layout()

    print("\n3D gradient map (projected onto YZ) generated.")


# ================================================================
# 3D Interactive Heatmap with Plotly
# ================================================================
def plot_3d_heatmap_plotly(counts3d, x_edges, y_edges, z_edges):
    """
    Create an interactive 3D volume visualization of absorption distribution using Plotly.
    Shows the absorption density throughout the 3D space.
    """
    print("\n=== GENERATING 3D INTERACTIVE HEATMAP ===")
    
    # Get bin centers
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2
    z_centers = (z_edges[:-1] + z_edges[1:]) / 2
    
    # Create meshgrid
    X, Y, Z = np.meshgrid(x_centers, y_centers, z_centers, indexing='ij')
    
    # Flatten for plotting
    x_flat = X.flatten()
    y_flat = Y.flatten()
    z_flat = Z.flatten()
    values_flat = counts3d.flatten()
    
    # Filter out zero values for cleaner visualization
    nonzero_mask = values_flat > 0
    x_plot = x_flat[nonzero_mask]
    y_plot = y_flat[nonzero_mask]
    z_plot = z_flat[nonzero_mask]
    values_plot = values_flat[nonzero_mask]
    
    # Further filter: only show points above 5% of maximum
    threshold = 0.05 * values_plot.max()
    significant_mask = values_plot >= threshold
    x_plot = x_plot[significant_mask]
    y_plot = y_plot[significant_mask]
    z_plot = z_plot[significant_mask]
    values_plot = values_plot[significant_mask]
    
    print(f"Plotting {len(values_plot)} voxels above 5% threshold ({threshold:.1f} rays)")
    print(f"Value range: [{values_plot.min():.1f}, {values_plot.max():.1f}]")
    
    # Normalize values for better marker sizing
    values_normalized = (values_plot - values_plot.min()) / (values_plot.max() - values_plot.min() + 1e-10)
    marker_sizes = values_normalized * 8  # Size range: 2 to 10
    
    # Create 3D scatter plot
    fig = go.Figure(data=go.Scatter3d(
        x=x_plot,
        y=y_plot,
        z=z_plot,
        mode='markers',
        marker=dict(
            size=marker_sizes,
            color=values_plot,
            colorscale='Turbo',
            colorbar=dict(title="Absorbed Rays"),
            opacity=0.8,
            line=dict(width=0)
        ),
        text=[f'Rays: {v:.1f}' for v in values_plot],
        hoverinfo='text+x+y+z'
    ))
    
    fig.update_layout(
        title="3D Absorption Distribution",
        scene=dict(
            xaxis_title="X position (mm)",
            yaxis_title="Y position (mm)",
            zaxis_title="Z position (mm)",
            xaxis=dict(range=[x_edges[0], x_edges[-1]]),
            yaxis=dict(range=[y_edges[0], y_edges[-1]]),
            zaxis=dict(range=[z_edges[0], z_edges[-1]]),
            aspectmode='data'
        ),
        width=900,
        height=800,
    )
    
    fig.show()
    print("3D interactive heatmap displayed in browser.")


# ================================================================
# Main Function
# ================================================================
def plot_saved_data(data_folder):
    """
    Load and analyze absorption data.
    
    Args:
        data_folder: Primary dataset folder path
        comparison_mode: If True, also process comparison dataset and plot curves together
    """
    data_path = Path(data_folder)
    csv_path = data_path / "absorbed_rays_raw_data.csv"

    if not csv_path.exists():
        print(f"Error: CSV file not found: {csv_path}")
        return

    print(f"\n=== LOADING DATA FROM: {csv_path} ===\n")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} absorbed rays from main file")


    xpos = df["x_position"].values
    ypos = df["y_position"].values
    zpos = df["z_position"].values
    dx = df["direction_x"].values
    dy = df["direction_y"].values
    dz = df["direction_z"].values


    # ---------- 3D histogram for primary dataset ----------
    counts3d_1, x_edges1, y_edges1, z_edges1 = make_3d_histogram(xpos, ypos, zpos)

    # 2D heatmap projection + 3D uniformity
    mask3d_1, U3d_1, CV3d_1 = plot_heatmap_3d_projection(
        counts3d_1, x_edges1, y_edges1, z_edges1, ENERGY_THRESHOLD_PERCENT
    )

    # 3D gradient visualization (projected)
    plot_gradient_map(counts3d_1, x_edges1, y_edges1, z_edges1)
    
    # 3D interactive heatmap with Plotly
    plot_3d_heatmap_plotly(counts3d_1, x_edges1, y_edges1, z_edges1)
    
    print("\n=== ALL PLOTS GENERATED ===")
    plt.show()




# ================================================================
# Entry Point
# ================================================================
if __name__ == "__main__":

    print("\nUsing default path:")
    print(DATA_PATH_DEFAULT)

    plot_saved_data(DATA_PATH_DEFAULT)
