import numpy as np
from pvtrace.geometry.utils import (
    flip,
    angle_between
)
# Fresnel

def fresnel_reflectivity(angle, n1, n2):
    # Catch TIR case
    if n2 < n1 and angle > np.arcsin(n2/n1):
        return 1.0
    c = np.cos(angle)
    s = np.sin(angle)
    k = np.sqrt(1 - (n1/n2 * s)**2)
    Rs1 = n1 * c - n2 * k
    Rs2 = n1 * c + n2 * k
    Rs = (Rs1/Rs2)**2
    Rp1 = n1 * k - n2 * c
    Rp2 = n1 * k + n2 * c
    Rp = (Rp1/Rp2)**2
    r = 0.5 * (Rs + Rp)
    return r

def specular_reflection(direction, normal):
    direction = np.array(direction)
    normal = np.array(normal)
    if np.dot(normal, direction) < 0.0:
        normal = flip(normal)
    d = np.dot(normal, direction)
    reflected_direction = direction - 2 * d * normal
    return reflected_direction


def fresnel_refraction(direction, normal, n1, n2):
    vector = np.array(direction)
    normal = np.array(normal)
    n = n1/n2
    dot = np.dot(vector, normal)
    c = np.sqrt(1 - n**2 * (1 - dot**2))
    sign = 1
    if dot < 0.0:
        sign = -1
    refracted_direction = n * vector + sign*(c - sign*n*dot) * normal
    return refracted_direction


# Lineshape

gaussian = lambda x, c1, c2, c3: c1*np.exp(-((c2 - x)/c3)**2)

bandgap = lambda x, cutoff, alpha: (1 - np.heaviside(x-cutoff, 0.5)) * alpha

def simple_convert_spectum(spec):
    """ Change spectrum x-axis only.
    """
    h = 6.62607015e-34  # J s
    c = 299792458.0     # m s-1
    q = 1.60217662e-19  # C
    kb = 1.38064852e-23 # K K-1
    conversion_constant = (h * c / q * 1e9)
    xp = conversion_constant / spec[:, 0]
    _spec = np.array(spec)
    _spec[:, 0] = xp
    return _spec


def thermodynamic_emission(abs_spec, T=300, mu=0.5):
    h = 6.62607015e-34  # J s
    c = 299792458.0     # m s-1
    q = 1.60217662e-19  # C
    kb = 1.38064852e-23 # J K-1
    conversion_constant = (h * c / q * 1e9)
    energy_spec = simple_convert_spectum(abs_spec)
    x, y = energy_spec[:, 0], energy_spec[:, 1]
    ems_energy = y * 2 * x**2 / (c**2 * (h/q)**3) / np.expm1((x-mu)/((kb/q)*T))
    ems_energy /= np.max(ems_energy)
    ems_wavelength = simple_convert_spectum(np.column_stack((x, ems_energy)))
    return ems_wavelength


# Coordinates

def spherical_to_cart(theta, phi, r=1):
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    cart = np.column_stack((x, y, z))
    if cart.size == 3:
        return cart[0,:]
    return cart

#  Volume scattering

def isotropic(theta_max=np.pi/2):
    """ Isotropic phase function.
    """
    g1, g2 = np.random.uniform(0, 1, 2)
    phi = 2 * np.pi * g1
    cos_theta_max = np.cos(theta_max)
    mu = (1 - cos_theta_max) * g2 + cos_theta_max  # uniform in mu
    theta = np.arccos(mu)
    coords = spherical_to_cart(theta, phi)
    return coords

def henyey_greenstein(g=0.0):
    """ Henyey-Greenstein phase function.
    """
    # https://www.astro.umd.edu/~jph/HG_note.pdf
    # Inverse is not defined at g=0 but in the limit
    # tends to the isotropic case.
    if np.isclose(g, 0.0):
        return isotropic()
    p = np.random.uniform(0, 1)
    s = 2 * p - 1
    mu = 1/(2*g) * (1 + g**2 - ((1 - g**2)/(1 + g*s))**2)
    phi = 2 * np.pi * np.random.uniform()
    theta = np.arccos(mu)
    coords = spherical_to_cart(theta, phi)
    return coords


# Light source /surface scattering

def cone(theta_max=np.pi/2):
    """ Samples directions within a cone of solid angle defined by `theta_max`.

        Notes
        -----
        Derived as follows using sympy::

            from sympy import *
            theta, theta_max, p = symbols('theta theta_max p')
            f = cos(theta) * sin(theta)
            cdf = integrate(f, (theta, 0, theta))
            pdf = cdf / cdf.subs({theta: theta_max})
            inv_pdf = solve(Eq(pdf, p), theta)[-1]
    """
    if np.isclose(theta_max, 0.0) or theta_max > np.pi/2:
        raise ValueError("Expected 0 < theta_max <= pi/2")
    p1, p2 = np.random.uniform(0, 1, 2)
    theta = np.arcsin(np.sqrt(p1) * np.sin(theta_max))
    phi = 2 * np.pi * p2
    coords = spherical_to_cart(theta, phi)
    return coords

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



def isotropic_scattering():
    return henyey_greenstein(-0.9)

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

# Global cache for CDF lookup table
_fresnel_cdf_cache = {}

def _build_fresnel_cdf(n1=1.6, n2=1.815, num_bins=100000):
    """
    Build cumulative distribution function (CDF) for Fresnel-corrected Lambertian.
    This is cached globally for speed.
    
    Returns:
    --------
    theta_n2_bins : np.ndarray
        Polar angles in medium2
    cdf : np.ndarray
        Cumulative distribution function values
    """
    cache_key = (n1, n2, num_bins)
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
    
    # Total transmission probability
    T_total = T_in * T_out
    
    # Energy distribution: dE/dθ = I(θ) * sin(θ) * 2π
    # For Lambertian: I(θ) ∝ cos(θ) * T_total
    # Energy per unit angle in medium2: E(θ) = n2^2 * cos(θ_n2) * T_total * sin(θ_n2)
    energy_distribution = n2**2 * np.cos(theta_n2) * T_total * np.sin(theta_n2)
    
    # Build CDF using cumulative trapezoid integration
    cdf = np.zeros_like(energy_distribution)
    cdf[1:] = np.cumsum(0.5 * (energy_distribution[1:] + energy_distribution[:-1]) * np.diff(theta_n2))
    
    # Normalize CDF to [0, 1]
    if cdf[-1] > 0:
        cdf /= cdf[-1]
    else:
        raise ValueError("CDF normalization failed - no transmitted energy")
    
    # Cache the result
    _fresnel_cdf_cache[cache_key] = (theta_n2, cdf)
    
    return theta_n2, cdf

def lambertian_with_fresnel(n1=1.6, n2=1.815, num_bins=100000):
    """
    Sample truncated Lambertian distribution with Fresnel transmission losses.
    Light passes through: air (n=1.0) → medium1 (n=n1) → medium2 (n=n2)
    
    Uses inverse transform sampling with pre-computed CDF for speed.
    First call builds lookup table (cached), subsequent calls are very fast.
    
    Parameters:
    -----------
    n1 : float
        Refractive index of first medium (default: 1.6)
    n2 : float
        Refractive index of second medium (default: 1.815)
    num_bins : int
        Number of bins for CDF lookup table (default: 10000)
        Higher = more accurate but slower first call
    
    Returns:
    --------
    coords : np.ndarray
        Unit direction vector [x, y, z] in medium2 coordinates
    
    Example:
    --------
    >>> direction = lambertian_with_fresnel(n1=1.6, n2=1.815)
    >>> print(direction)  # [x, y, z] unit vector
    """
    # Build or retrieve cached CDF
    theta_n2_bins, cdf = _build_fresnel_cdf(n1, n2, num_bins)
    
    # Sample uniformly from [0, 1]
    u = np.random.uniform(0, 1)
    
    # Inverse transform sampling: find theta_n2 where CDF(theta_n2) = u
    # Use linear interpolation for speed
    theta_n2 = np.interp(u, cdf, theta_n2_bins)
    
    # Sample azimuthal angle uniformly
    phi = 2 * np.pi * np.random.uniform(0, 1)
    
    # Convert to Cartesian coordinates
    coords = spherical_to_cart(theta_n2, phi)
    return coords