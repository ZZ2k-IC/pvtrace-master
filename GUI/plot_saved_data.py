"""
Plot Saved Ray Tracing Data
This script reads saved CSV data from a simulation folder and reproduces all plots
without needing to re-run the simulation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# ========== CONFIGURATION ==========
THRESHOLD_PERCENTAGE = 0.1  # 36.8% (1/e criterion) - Change this value as needed
HEATMAP_BINS_Y = 75  # Number of bins along Y-axis
HEATMAP_BINS_Z = 100  # Number of bins along Z-axis
# ===================================
# Plot range configuration (set to None for auto-range)
Y_RANGE = (-3, 3)  # e.g., (-3, 3) to show Y from -3 to 3 mm, or None for auto
Z_RANGE = (0, 8)  # e.g., (0, 17) to show Z from 0 to 17 mm, or None for auto
# ===================================
def plot_saved_data(data_folder):
    """
    Read saved CSV files and generate all visualization plots
    
    Parameters:
    -----------
    data_folder : str
        Path to folder containing saved CSV files
    """
    data_path = Path(data_folder)
    
    if not data_path.exists():
        print(f"Error: Folder {data_folder} does not exist!")
        return
    
    print(f"\n=== LOADING DATA FROM: {data_folder} ===\n")
    
    # Load absorbed ray raw data
    absorbed_csv = data_path / "absorbed_rays_raw_data.csv"
    if not absorbed_csv.exists():
        print(f"Error: {absorbed_csv} not found!")
        return
    
    absorbed_data = pd.read_csv(absorbed_csv)
    print(f"Loaded {len(absorbed_data)} absorbed rays")
    
    # Extract position and direction data
    xpos_abs = absorbed_data['x_position'].values
    ypos_abs = absorbed_data['y_position'].values
    zpos_abs = absorbed_data['z_position'].values
    absorbed_wavs = absorbed_data['wavelength'].values
    direction_x = absorbed_data['direction_x'].values
    direction_y = absorbed_data['direction_y'].values
    direction_z = absorbed_data['direction_z'].values
    
    num_rays = len(absorbed_data)
    
    # ========== FIGURE 1: YZ HEATMAP WITH CONTOUR (USING RAW DATA) ==========
    plt.figure(1, figsize=(12, 8), clear=True)
    
    if len(absorbed_data) > 0:
        # Determine ranges for histogram
        if Y_RANGE is None:
            y_range = [ypos_abs.min(), ypos_abs.max()]
        else:
            y_range = Y_RANGE
        
        if Z_RANGE is None:
            z_range = [zpos_abs.min(), zpos_abs.max()]
        else:
            z_range = Z_RANGE
        
        print(f"Y range: {y_range[0]:.3f} to {y_range[1]:.3f} mm")
        print(f"Z range: {z_range[0]:.3f} to {z_range[1]:.3f} mm")
        
        # Create 2D histogram from raw data with specified range (Z on X-axis, Y on Y-axis)
        counts, z_edges, y_edges = np.histogram2d(
            zpos_abs, ypos_abs,  # SWAPPED: Z first, Y second
            bins=[HEATMAP_BINS_Z, HEATMAP_BINS_Y],  # SWAPPED bins
            range=[z_range, y_range]  # SWAPPED ranges
        )
        # Calculate effective absorbed area using configurable threshold
        max_count = np.max(counts)
        threshold = max_count * THRESHOLD_PERCENTAGE
        
        # Find bins that exceed the threshold
        effective_mask = counts >= threshold
        effective_bins = np.sum(effective_mask)
        
        # Calculate bin area
        z_bin_width = (z_edges[1] - z_edges[0])
        y_bin_width = (y_edges[1] - y_edges[0])
        bin_area = z_bin_width * y_bin_width  # mm²
        
        # Calculate effective absorbed area
        effective_absorbed_area = effective_bins * bin_area
        
        # Calculate total area covered by any absorption
        total_mask = counts > 0
        total_bins = np.sum(total_mask)
        total_absorbed_area = total_bins * bin_area
        
        # Calculate area efficiency
        area_efficiency = (effective_absorbed_area / total_absorbed_area * 100) if total_absorbed_area > 0 else 0
        
        # Print results
        print(f"\n=== ABSORPTION AREA ANALYSIS ===")
        print(f"Threshold setting: {THRESHOLD_PERCENTAGE*100:.1f}%")
        print(f"Maximum absorption density: {max_count} rays/bin")
        print(f"Threshold value: {threshold:.1f} rays/bin")
        print(f"Effective absorbed area: {effective_absorbed_area:.3f} mm²")
        print(f"Total absorbed area: {total_absorbed_area:.3f} mm²")
        print(f"Area efficiency: {area_efficiency:.1f}%")
        print(f"Bin size: {z_bin_width:.4f} × {y_bin_width:.4f} mm")
        print(f"Grid resolution: {len(z_edges)-1} × {len(y_edges)-1} bins")
        
        # HEATMAP (Z on X-axis, Y on Y-axis)
        plt.imshow(
            counts.T,
            origin='lower',
            extent=[z_edges[0], z_edges[-1], y_edges[0], y_edges[-1]],  # SWAPPED extent
            cmap='viridis',
            aspect='equal',  # Changed from 'auto' to 'equal' for same scale
            interpolation='nearest'
        )
        
        cbar = plt.colorbar(label='Number of absorbed rays')
        
        # Replace the entire contour section (Z on X-axis, Y on Y-axis):
        if threshold > 0:
            Z_centers = (z_edges[:-1] + z_edges[1:]) / 2  # SWAPPED
            Y_centers = (y_edges[:-1] + y_edges[1:]) / 2  # SWAPPED
            Z, Y = np.meshgrid(Z_centers, Y_centers)  # SWAPPED order
            
            # Draw FILLED contour for threshold region (semi-transparent)
            plt.contourf(
                Z, Y, counts.T,  # SWAPPED Z and Y
                levels=[threshold, max_count],
                colors=['red'],
                alpha=0.1  # Semi-transparent fill
            )
            
            # Draw THICK boundary line at threshold
            threshold_contour = plt.contour(
                Z, Y, counts.T,  # SWAPPED Z and Y
                levels=[threshold], 
                colors='red', 
                linewidths=1,
                linestyles='solid'
            )
            

            
        plt.title(f'Light absorption density on ZY plane ({num_rays} rays absorbed)')
        plt.xlabel('Z position (mm)')  # SWAPPED: Z on X-axis
        plt.ylabel('Y position (mm)')  # SWAPPED: Y on Y-axis
        plt.grid(False)
        
        # Create custom legend with only two items
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch
        
        legend_elements = [
            Line2D([0], [0], color='red', linewidth=2, linestyle='solid', 
                   label=f'Threshold line ({THRESHOLD_PERCENTAGE*100:.0f}%)'),            
        ]
        
        plt.legend(handles=legend_elements, loc='upper right', 
                  frameon=True, fancybox=True, shadow=True)
    
    plt.tight_layout()
    
    # ========== FIGURE 7: Z-AXIS HISTOGRAM ==========
    plt.figure(7, clear=True)
    if len(zpos_abs) > 0:
        plt.hist(zpos_abs, bins=50, range=(0, 17), alpha=0.7, color='blue', edgecolor='black')
        plt.title(f'Absorbed rays distribution along Z-axis ({num_rays} rays)')
        plt.xlabel('Z position (mm)')
        plt.ylabel('Number of absorbed rays')
        plt.grid(True, alpha=0.3)
        plt.xlim(0, 17)
        
        z_mean = np.mean(zpos_abs)
        z_std = np.std(zpos_abs)
        plt.text(0.02, 0.98, 
                f'Mean Z: {z_mean:.2f} mm\nStd Z: {z_std:.2f} mm\nTotal: {len(zpos_abs)} rays', 
                transform=plt.gca().transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # ========== FIGURE 8: Y-AXIS HISTOGRAM ==========
    plt.figure(8, clear=True)
    if len(ypos_abs) > 0:
        plt.hist(ypos_abs, bins=50, range=(-3, 3), alpha=0.7, color='green', edgecolor='black')
        plt.title(f'Absorbed rays distribution along Y-axis ({num_rays} rays)')
        plt.xlabel('Y position (mm)')
        plt.ylabel('Number of absorbed rays')
        plt.grid(True, alpha=0.3)
        
        y_mean = np.mean(ypos_abs)
        y_std = np.std(ypos_abs)
        plt.text(0.02, 0.98, 
                f'Mean Y: {y_mean:.2f} mm\nStd Y: {y_std:.2f} mm\nTotal: {len(ypos_abs)} rays', 
                transform=plt.gca().transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # ========== FIGURE 9: X-AXIS HISTOGRAM ==========
    plt.figure(9, clear=True)
    if len(xpos_abs) > 0:
        plt.hist(xpos_abs, bins=50, range=(-2.5, 2.5), alpha=0.7, color='red', edgecolor='black')
        plt.title(f'Absorbed rays distribution along X-axis ({num_rays} rays)')
        plt.xlabel('X position (mm)')
        plt.ylabel('Number of absorbed rays')
        plt.grid(True, alpha=0.3)
        plt.xlim(-2.5, 2.5)
        
        x_mean = np.mean(xpos_abs)
        x_std = np.std(xpos_abs)
        plt.text(0.02, 0.98, 
                f'Mean X: {x_mean:.2f} mm\nStd X: {x_std:.2f} mm\nTotal: {len(xpos_abs)} rays', 
                transform=plt.gca().transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # ========== FIGURE 10: AZIMUTHAL ANGLE HISTOGRAM ==========
    plt.figure(10, clear=True)
    if len(direction_x) > 0:
        azimuthal_angles = []
        for i in range(len(direction_x)):
            azimuthal_angle = np.degrees(np.arctan2(direction_y[i], direction_x[i]))
            if azimuthal_angle < 0:
                azimuthal_angle += 360
            azimuthal_angles.append(azimuthal_angle)
        
        plt.hist(azimuthal_angles, bins=36, range=(0, 360), alpha=0.7, color='purple', edgecolor='black')
        plt.title(f'Absorbed rays azimuthal angle distribution ({num_rays} rays)')
        plt.xlabel('Azimuthal angle (degrees)')
        plt.ylabel('Number of absorbed rays')
        plt.grid(True, alpha=0.3)
        plt.xlim(0, 360)
        
        az_mean = np.mean(azimuthal_angles)
        az_std = np.std(azimuthal_angles)
        plt.text(0.02, 0.98, 
                f'Mean: {az_mean:.1f}°\nStd: {az_std:.1f}°\nTotal: {len(azimuthal_angles)} rays', 
                transform=plt.gca().transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # ========== FIGURE 11: POLAR ANGLE HISTOGRAM ==========
    plt.figure(11, clear=True)
    if len(direction_z) > 0:
        polar_angles = []
        for i in range(len(direction_z)):
            cos_theta = abs(direction_z[i])
            cos_theta = np.clip(cos_theta, 0, 1)
            theta_rad = np.arccos(cos_theta)
            theta_deg = np.degrees(theta_rad)
            polar_angles.append(theta_deg)
        
        plt.hist(polar_angles, bins=90, range=(0, 90), alpha=0.7, color='orange', edgecolor='black')
        plt.title(f'Absorbed rays polar angle distribution ({num_rays} rays)')
        plt.xlabel('Polar angle (degrees)')
        plt.ylabel('Number of absorbed rays')
        plt.grid(True, alpha=0.3)
        plt.xlim(0, 90)
        
        pol_mean = np.mean(polar_angles)
        pol_std = np.std(polar_angles)
        plt.text(0.02, 0.98, 
                f'Mean: {pol_mean:.1f}°\nStd: {pol_std:.1f}°\nTotal: {len(polar_angles)} rays', 
                transform=plt.gca().transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    print("\n=== ALL PLOTS GENERATED ===")
    print("Close all plot windows to exit.")
    plt.show()


if __name__ == "__main__":
    # You can either provide the path as a command line argument or edit it here
    if len(sys.argv) > 1:
        folder_path = sys.argv[1]
    else:
        # EDIT THIS PATH to point to your saved data folder
        folder_path = r"C:\Users\Zedd\OneDrive - Imperial College London\UROP\Big_data"
        
        print("Usage: python plot_saved_data.py <path_to_data_folder>")
        print(f"Using default path: {folder_path}")
        print("(You can change the default path in the script or provide it as an argument)\n")
    
    plot_saved_data(folder_path)
