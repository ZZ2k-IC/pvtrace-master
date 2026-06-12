#%%
# Python code to compute and plot the angular distributions:
# - I0(theta): original Lambertian in air (I0 = cos(theta))
# - I_final(theta): after passing air -> slab(n) -> air (single pass, parallel faces, no multiple reflections)
#   with Fresnel transmission at both interfaces (unpolarized average).
# This visualizes how Fresnel modulation changes the Lambertian shape.
import numpy as np
import matplotlib.pyplot as plt

def fresnel_Rs_Rp(n1, n2, theta_i):
    """
    Return (Rs, Rp) reflectances for intensity for incidence from medium n1 at angle theta_i (radians).
    Handles total internal reflection when applicable.
    """
    # clamp domain
    sin_theta_i = np.sin(theta_i)
    # compute sin(theta_t) via Snell; might be >1 indicating TIR if n1>n2
    sin_theta_t = n1 / n2 * sin_theta_i
    # For array support, handle elementwise
    Rs = np.zeros_like(theta_i, dtype=float)
    Rp = np.zeros_like(theta_i, dtype=float)
    # TIR mask
    tir_mask = np.abs(sin_theta_t) > 1.0
    # For TIR: reflectance = 1
    Rs[tir_mask] = 1.0
    Rp[tir_mask] = 1.0
    # For non-TIR: compute theta_t and Fresnel formulas
    non_tir = ~tir_mask
    if np.any(non_tir):
        theta_t = np.arcsin(np.clip(sin_theta_t[non_tir], -1.0, 1.0))
        ci = np.cos(theta_i[non_tir])
        ct = np.cos(theta_t)
        n1_loc = n1
        n2_loc = n2
        Rs_loc = ((n1_loc * ci - n2_loc * ct) / (n1_loc * ci + n2_loc * ct))**2
        Rp_loc = ((n1_loc * ct - n2_loc * ci) / (n1_loc * ct + n2_loc * ci))**2
        Rs[non_tir] = Rs_loc
        Rp[non_tir] = Rp_loc
    return Rs, Rp

def unpolarized_transmission(n1, n2, theta_i):
    """
    Return unpolarized intensity transmittance (fraction of incident power transmitted)
    for incidence from n1 at angle theta_i into n2.
    Uses T = 1 - (Rs+Rp)/2, with TIR handled (T=0).
    """
    Rs, Rp = fresnel_Rs_Rp(n1, n2, theta_i)
    R_unpol = 0.5 * (Rs + Rp)
    T_unpol = 1.0 - R_unpol
    # For TIR, ensure T=0
    # (fresnel_Rs_Rp already set Rs=Rp=1 in TIR mask, so T_unpol will be 0 there)
    return T_unpol

# Parameters
n1 = 1.63  # refractive index of first medium
n2 = 1.45  # refractive index of second medium
n3 = 1.6   # refractive index of third medium
theta_deg = np.linspace(0, 89.99, 1000)  # avoid exactly 90 deg
theta = np.deg2rad(theta_deg)

# Original Lambertian in air (unit prefactor)
I0 = np.cos(theta)  # Lambertian intensity ~ cos(theta) per unit solid angle

# Transmission at first interface (air -> n1)
T_in = unpolarized_transmission(1.0, n1, theta)
# Compute theta inside medium1 via Snell
sin_theta_in_n1 = np.sin(theta) / n1
sin_theta_in_n1 = np.clip(sin_theta_in_n1, -1.0, 1.0)
theta_in_n1 = np.arcsin(sin_theta_in_n1)

# Transmission at second interface (n1 -> n2) evaluated at internal angle in n1
T_out = unpolarized_transmission(n1, n2, theta_in_n1)
# Compute theta inside medium2 via Snell
sin_theta_in_n2 = n1 * np.sin(theta_in_n1) / n2
sin_theta_in_n2 = np.clip(sin_theta_in_n2, -1.0, 1.0)
theta_in_n2 = np.arcsin(sin_theta_in_n2)

# Transmission at third interface (n2 -> n3) evaluated at internal angle in n2
T_out2 = unpolarized_transmission(n2, n3, theta_in_n2)
# Compute theta inside medium3 via Snell
sin_theta_in_n3 = n2 * np.sin(theta_in_n2) / n3
sin_theta_in_n3 = np.clip(sin_theta_in_n3, -1.0, 1.0)
theta_in_n3 = np.arcsin(sin_theta_in_n3)

# Total single-pass transmittance (neglecting multiple internal reflections)
T_total = T_in * T_out * T_out2


# Plot 6 diagrams (2x3)
plt.figure(figsize=(20,10))


# (A) Inside medium1: ideal truncated Lambertian vs actual (with Fresnel) - INTENSITY
I_inside_n1 = n1**2 * np.cos(theta_in_n1) * T_in
theta1_deg = np.rad2deg(theta_in_n1)
theta_crit_n1 = np.arcsin(1/n1)
mask_n1 = theta_in_n1 <= theta_crit_n1

plt.subplot(2,2,1)
plt.plot(theta1_deg[mask_n1], n1**2*np.cos(theta_in_n1[mask_n1]), '--', label='Ideal truncated Lambertian (no Fresnel)')
plt.plot(theta1_deg[mask_n1], I_inside_n1[mask_n1], label='Actual in medium1 (with Fresnel)')
plt.xlabel('Internal polar angle θ₁ (deg)')
plt.ylabel('Intensity (a.u.)')
plt.title(f'A) Intensity in medium1 (n={n1})')
plt.legend()
plt.grid(True)

# (B) Inside medium2: ideal truncated Lambertian vs actual (with Fresnel at both interfaces) - INTENSITY
I_inside_n2 = n2**2 * np.cos(theta_in_n2) * (T_in * T_out)
theta2_deg = np.rad2deg(theta_in_n2)
# Critical angle at n1->n2 interface
theta_crit_n1_to_n2 = np.arcsin(min(1.0, n2/n1))
mask_n2 = theta_in_n1 <= theta_crit_n1_to_n2

plt.subplot(2,2,2)
plt.plot(theta2_deg[mask_n2], n2**2*np.cos(theta_in_n2[mask_n2]), '--', label='Ideal truncated Lambertian (no Fresnel)')
plt.plot(theta2_deg[mask_n2], I_inside_n2[mask_n2], label='Actual in medium2 (with Fresnel)')
plt.xlabel('Internal polar angle θ₂ (deg)')
plt.ylabel('Intensity (a.u.)')
plt.title(f'B) Intensity in medium2 (n={n2})')
plt.legend()
plt.grid(True)

# (C) Energy distribution in medium1 (dE/dθ = I * sin(θ) * 2π)
# Energy per unit polar angle: dE/dθ = I(θ) * 2π * sin(θ)
E_ideal_n1 = n1**2 * np.cos(theta_in_n1) * np.sin(theta_in_n1) * 2 * np.pi
E_actual_n1 = I_inside_n1 * np.sin(theta_in_n1) * 2 * np.pi

plt.subplot(2,2,3)
plt.plot(theta1_deg[mask_n1], E_ideal_n1[mask_n1], '--', label='Ideal truncated Lambertian (no Fresnel)')
plt.plot(theta1_deg[mask_n1], E_actual_n1[mask_n1], label='Actual in medium1 (with Fresnel)')
plt.xlabel('Internal polar angle θ₁ (deg)')
plt.ylabel('Energy per unit angle (a.u.)')
plt.title(f'C) Energy distribution in medium1 (n={n1})')
plt.legend()
plt.grid(True)

# Print integrated energies for medium1
E_total_ideal_n1 = np.trapz(E_ideal_n1[mask_n1], theta_in_n1[mask_n1])
E_total_actual_n1 = np.trapz(E_actual_n1[mask_n1], theta_in_n1[mask_n1])

# Calculate original Lambertian energy in air for comparison
E_original_air = np.trapz(I0 * np.sin(theta) * 2 * np.pi, theta)

print(f"Original Lambertian energy in air: {E_original_air:.4f}")
print(f"Medium1 - Total energy (ideal): {E_total_ideal_n1:.4f}")
print(f"Medium1 - Total energy (actual): {E_total_actual_n1:.4f}")
print(f"Medium1 - Transmission efficiency (vs ideal): {E_total_actual_n1/E_total_ideal_n1*100:.2f}%")
print(f"Medium1 - Total energy percentage (vs original): {E_total_actual_n1/E_original_air*100:.2f}%")

# (D) Energy distribution in medium2 (dE/dθ = I * sin(θ) * 2π)
E_ideal_n2 = n2**2 * np.cos(theta_in_n2) * np.sin(theta_in_n2) * 2 * np.pi
E_actual_n2 = I_inside_n2 * np.sin(theta_in_n2) * 2 * np.pi

plt.subplot(2,2,4)
plt.plot(theta2_deg[mask_n2], E_ideal_n2[mask_n2], '--', label='Ideal truncated Lambertian (no Fresnel)')
plt.plot(theta2_deg[mask_n2], E_actual_n2[mask_n2], label='Actual in medium2 (with Fresnel)')
plt.xlabel('Internal polar angle θ₂ (deg)')
plt.ylabel('Energy per unit angle (a.u.)')
plt.title(f'D) Energy distribution in medium2 (n={n2})')
plt.legend()
plt.grid(True)

# Print integrated energies for medium2
E_total_ideal_n2 = np.trapz(E_ideal_n2[mask_n2], theta_in_n2[mask_n2])
E_total_actual_n2 = np.trapz(E_actual_n2[mask_n2], theta_in_n2[mask_n2])
print(f"\nMedium2 - Total energy (ideal): {E_total_ideal_n2:.4f}")
print(f"Medium2 - Total energy (actual): {E_total_actual_n2:.4f}")
print(f"Medium2 - Transmission efficiency (vs ideal): {E_total_actual_n2/E_total_ideal_n2*100:.2f}%")
print(f"Medium2 - Total energy percentage (vs original): {E_total_actual_n2/E_original_air*100:.2f}%")

# (E) Inside medium3: ideal truncated Lambertian vs actual (with Fresnel at all interfaces) - INTENSITY
I_inside_n3 = n3**2 * np.cos(theta_in_n3) * (T_in * T_out * T_out2)
theta3_deg = np.rad2deg(theta_in_n3)
# Critical angle at n2->n3 interface
theta_crit_n2_to_n3 = np.arcsin(min(1.0, n3/n2))
mask_n3 = theta_in_n2 <= theta_crit_n2_to_n3

plt.subplot(2,3,5)
plt.plot(theta3_deg[mask_n3], n3**2*np.cos(theta_in_n3[mask_n3]), '--', label='Ideal truncated Lambertian (no Fresnel)')
plt.plot(theta3_deg[mask_n3], I_inside_n3[mask_n3], label='Actual in medium3 (with Fresnel)')
plt.xlabel('Internal polar angle θ₃ (deg)')
plt.ylabel('Intensity (a.u.)')
plt.title(f'E) Intensity in medium3 (n={n3})')
plt.legend()
plt.grid(True)

# (F) Energy distribution in medium3 (dE/dθ = I * sin(θ) * 2π)
E_ideal_n3 = n3**2 * np.cos(theta_in_n3) * np.sin(theta_in_n3) * 2 * np.pi
E_actual_n3 = I_inside_n3 * np.sin(theta_in_n3) * 2 * np.pi

plt.subplot(2,3,6)
plt.plot(theta3_deg[mask_n3], E_ideal_n3[mask_n3], '--', label='Ideal truncated Lambertian (no Fresnel)')
plt.plot(theta3_deg[mask_n3], E_actual_n3[mask_n3], label='Actual in medium3 (with Fresnel)')
plt.xlabel('Internal polar angle θ₃ (deg)')
plt.ylabel('Energy per unit angle (a.u.)')
plt.title(f'F) Energy distribution in medium3 (n={n3})')
plt.legend()
plt.grid(True)

# Print integrated energies for medium3
E_total_ideal_n3 = np.trapz(E_ideal_n3[mask_n3], theta_in_n3[mask_n3])
E_total_actual_n3 = np.trapz(E_actual_n3[mask_n3], theta_in_n3[mask_n3])
print(f"\nMedium3 - Total energy (ideal): {E_total_ideal_n3:.4f}")
print(f"Medium3 - Total energy (actual): {E_total_actual_n3:.4f}")
print(f"Medium3 - Transmission efficiency (vs ideal): {E_total_actual_n3/E_total_ideal_n3*100:.2f}%")
print(f"Medium3 - Total energy percentage (vs original): {E_total_actual_n3/E_original_air*100:.2f}%")
print(f"\nOverall transmission through all three interfaces: {E_total_actual_n3/E_original_air*100:.2f}%")

plt.tight_layout()
plt.show()
#%%
def spherical_to_cart(theta, phi):
    """
    Convert spherical coordinates (theta, phi) to Cartesian (x, y, z).
    theta: polar angle from z-axis
    phi: azimuthal angle
    """
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return np.array([x, y, z])

def lambertian(theta_max=np.pi/2):
    """
    Sample Lambertian distribution limited to theta_max.
    """
    if not (0 < theta_max <= np.pi / 2):
        raise ValueError("theta_max must be in (0, pi/2]")

    p1, p2 = np.random.uniform(0, 1, 2)
    theta = np.arcsin(np.sqrt(p1) * np.sin(theta_max))  # corrected sampling
    phi = 2 * np.pi * p2
    coords = spherical_to_cart(theta, phi)
    return coords

# Global cache for CDF lookup table
_fresnel_cdf_cache = {}

def _build_fresnel_cdf(n1=1.6, n2=1.815, n3=None, num_bins=1000000):
    """
    Build cumulative distribution function (CDF) for Fresnel-corrected Lambertian.
    This is cached globally for speed.
    
    Returns:
    --------
    theta_final_bins : np.ndarray
        Polar angles in final medium (n2 or n3 if provided)
    cdf : np.ndarray
        Cumulative distribution function values
    """
    cache_key = (n1, n2, n3, num_bins)
    if cache_key in _fresnel_cdf_cache:
        return _fresnel_cdf_cache[cache_key]
    
    # Create fine grid of angles in air
    theta_air = np.linspace(0, np.pi/2, num_bins)
    
    # Calculate transmission at air -> n1 interface
    T_in = unpolarized_transmission(1.0, n1, theta_air)
    
    # Refract into medium1
    sin_theta_n1 = np.sin(theta_air) / n1
    sin_theta_n1 = np.clip(sin_theta_n1, -1.0, 1.0)
    theta_n1 = np.arcsin(sin_theta_n1)
    
    # Critical angle for n1 -> n2 interface
    if n2 > n1:
        theta_crit_n1_to_n2 = np.pi / 2
    else:
        theta_crit_n1_to_n2 = np.arcsin(n2 / n1)
    
    # Calculate transmission at n1 -> n2 interface
    T_out = np.zeros_like(theta_n1)
    mask_transmit = theta_n1 <= theta_crit_n1_to_n2
    T_out[mask_transmit] = unpolarized_transmission(n1, n2, theta_n1[mask_transmit])
    
    # Refract into medium2
    sin_theta_n2 = n1 * np.sin(theta_n1) / n2
    sin_theta_n2 = np.clip(sin_theta_n2, -1.0, 1.0)
    theta_n2 = np.arcsin(sin_theta_n2)
    
    # Total transmission probability so far
    T_total = T_in * T_out
    
    # If third medium specified, continue transmission
    if n3 is not None:
        # Critical angle for n2 -> n3 interface
        if n3 > n2:
            theta_crit_n2_to_n3 = np.pi / 2
        else:
            theta_crit_n2_to_n3 = np.arcsin(n3 / n2)
        
        # Calculate transmission at n2 -> n3 interface
        T_out2 = np.zeros_like(theta_n2)
        mask_transmit2 = theta_n2 <= theta_crit_n2_to_n3
        T_out2[mask_transmit2] = unpolarized_transmission(n2, n3, theta_n2[mask_transmit2])
        
        # Refract into medium3
        sin_theta_n3 = n2 * np.sin(theta_n2) / n3
        sin_theta_n3 = np.clip(sin_theta_n3, -1.0, 1.0)
        theta_n3 = np.arcsin(sin_theta_n3)
        
        # Update total transmission
        T_total = T_total * T_out2
        
        # Use medium3 as final medium
        theta_final = theta_n3
        n_final = n3
    else:
        # Use medium2 as final medium
        theta_final = theta_n2
        n_final = n2
    
    # Energy distribution: dE/dθ = I(θ) * sin(θ) * 2π
    # For Lambertian: I(θ) ∝ cos(θ) * T_total
    # Energy per unit angle in final medium: E(θ) = n_final^2 * cos(θ_final) * T_total * sin(θ_final)
    energy_distribution = n_final**2 * np.cos(theta_final) * T_total * np.sin(theta_final)
    
    # Build CDF using cumulative trapezoid integration
    cdf = np.zeros_like(energy_distribution)
    cdf[1:] = np.cumsum(0.5 * (energy_distribution[1:] + energy_distribution[:-1]) * np.diff(theta_final))
    
    # Normalize CDF to [0, 1]
    if cdf[-1] > 0:
        cdf /= cdf[-1]
    else:
        raise ValueError("CDF normalization failed - no transmitted energy")
    
    # Cache the result
    _fresnel_cdf_cache[cache_key] = (theta_final, cdf)
    
    return theta_final, cdf

def lambertian_with_fresnel(n1=1.6, n2=1.815, n3=None, num_bins=1000000):
    """
    Sample truncated Lambertian distribution with Fresnel transmission losses.
    Light passes through: air (n=1.0) → medium1 (n=n1) → medium2 (n=n2) → [medium3 (n=n3)]
    
    Uses inverse transform sampling with pre-computed CDF for speed.
    First call builds lookup table (cached), subsequent calls are very fast.
    
    Parameters:
    -----------
    n1 : float
        Refractive index of first medium (default: 1.6)
    n2 : float
        Refractive index of second medium (default: 1.815)
    n3 : float or None
        Refractive index of third medium (optional, default: None)
    num_bins : int
        Number of bins for CDF lookup table (default: 10000)
        Higher = more accurate but slower first call
    
    Returns:
    --------
    coords : np.ndarray
        Unit direction vector [x, y, z] in final medium coordinates
    
    Example:
    --------
    >>> direction = lambertian_with_fresnel(n1=1.6, n2=1.815, n3=1.6)
    >>> print(direction)  # [x, y, z] unit vector
    """
    # Build or retrieve cached CDF
    theta_final_bins, cdf = _build_fresnel_cdf(n1, n2, n3, num_bins)
    
    # Sample uniformly from [0, 1]
    u = np.random.uniform(0, 1)
    
    # Inverse transform sampling: find theta_final where CDF(theta_final) = u
    # Use linear interpolation for speed
    theta_final = np.interp(u, cdf, theta_final_bins)
    
    # Sample azimuthal angle uniformly
    phi = 2 * np.pi * np.random.uniform(0, 1)
    
    # Convert to Cartesian coordinates
    coords = spherical_to_cart(theta_final, phi)
    return coords