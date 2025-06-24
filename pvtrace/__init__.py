"""
Optical ray tracing for luminescent materials and spectral converter photovoltaic devices
"""
# import logging
# logging.basicConfig(level=logging.DEBUG)
# logger = logging.getLogger('pvtrace')

# Import commonly used classes to pvtrace namespace so users don't
# have to understand module layout.

# algorithm
from .algorithm import photon_tracer

# data
from .data import lumogen_f_red_305

# geometry
from .geometry.box import Box
from .geometry.cylinder import Cylinder
from .geometry.mesh import Mesh
from .geometry.sphere import Sphere
from .geometry.plane import Plane
from .geometry.planar_detector import create_planar_detector_node


# light
from .light.light import Light, rectangular_mask, circular_mask, cube_mask
from .light.ray import Ray

# material
from .material.component import (
    Scatterer,
    Absorber,
    Luminophore,
)
from .material.distribution import Distribution
from .material.material import Material
from .material.surface import (
    Surface,
    SurfaceDelegate,
    NullSurfaceDelegate,
    FresnelSurfaceDelegate
)
from .material.utils import isotropic, henyey_greenstein, cone, lambertian

# scene
from .scene.node import Node
from .scene.scene import Scene
from .scene.renderer import MeshcatRenderer
from .device.lsc import LSC
