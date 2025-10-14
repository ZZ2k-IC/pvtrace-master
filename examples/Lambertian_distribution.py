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
n1 = 1.6  # refractive index of first medium
n2 = 1.81  # refractive index of second medium
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

# Total single-pass transmittance (neglecting multiple internal reflections)
T_total = T_in * T_out


# Plot 4 diagrams (2x2)
plt.figure(figsize=(16,10))


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
I_inside_n2 = n2**2 * np.cos(theta_in_n2) * T_total
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
print(f"Medium1 - Total energy (ideal): {E_total_ideal_n1:.4f}")
print(f"Medium1 - Total energy (actual): {E_total_actual_n1:.4f}")
print(f"Medium1 - Transmission efficiency: {E_total_actual_n1/E_total_ideal_n1*100:.2f}%")

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
print(f"Medium2 - Transmission efficiency: {E_total_actual_n2/E_total_ideal_n2*100:.2f}%")

plt.tight_layout()
plt.show()