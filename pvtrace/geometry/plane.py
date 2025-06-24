from pvtrace.geometry.geometry import Geometry
from pvtrace.geometry.utils import EPS_ZERO
from pvtrace.common.errors import GeometryError
import numpy as np
import logging

logger = logging.getLogger(__name__)


class Plane(Geometry):
    """ Defines a rectangular plane with center (0, 0, 0) and specified width/height.
    
        Notes
        -----
        This is a true geometric plane (zero thickness) implementation.
        The plane is oriented according to the normal vector.
    """

    def __init__(self, size, normal=(0, 0, 1), material=None):
        """ Parameters
            ----------
            size : tuple of float
                The dimensions of the plane like (width, height)
            normal : tuple of float  
                Normal vector of the plane (default: pointing in +z direction)
            material : Material
                The material of the plane
        """
        super(Plane, self).__init__()
        self._size = np.array(size[:2])  # Only width and height
        self._normal = np.array(normal, dtype=np.float64)
        self._normal = self._normal / np.linalg.norm(self._normal)  # Normalize
        self._material = material
        
        # Create basis vectors for the plane
        self._create_basis_vectors()
    
    def _create_basis_vectors(self):
        """ Create orthonormal basis vectors for the plane. """
        # Find a vector not parallel to normal
        if abs(self._normal[0]) < 0.9:
            temp = np.array([1, 0, 0])
        else:
            temp = np.array([0, 1, 0])
        
        # Create first basis vector (u direction)
        self._u = temp - np.dot(temp, self._normal) * self._normal
        self._u = self._u / np.linalg.norm(self._u)
        
        # Create second basis vector (v direction) 
        self._v = np.cross(self._normal, self._u)

    @property
    def material(self):
        return self._material

    @material.setter
    def material(self, new_value):
        self._material = new_value

    def contains(self, point):
        """ A plane has no volume, so nothing is 'inside' it. """
        return False

    def is_on_surface(self, point):
        """ Check if point lies on the plane surface. """
        point = np.array(point)
        
        # Check if point is on the plane (distance to plane is zero)
        distance_to_plane = np.dot(point, self._normal)
        if abs(distance_to_plane) > EPS_ZERO:
            return False
        
        # Check if point is within the rectangular bounds
        u_coord = np.dot(point, self._u)
        v_coord = np.dot(point, self._v)
        
        return (abs(u_coord) <= self._size[0]/2 + EPS_ZERO and 
                abs(v_coord) <= self._size[1]/2 + EPS_ZERO)

    def intersections(self, origin, direction):
        """ Find intersection of ray with plane. """
        # Check for zero direction vector
        if direction == (0.0, 0.0, 0.0):
            return []
            
        origin = np.array(origin)
        direction = np.array(direction)
        
        # Ray-plane intersection: t = -dot(origin, normal) / dot(direction, normal)
        denominator = np.dot(direction, self._normal)
        
        if abs(denominator) < EPS_ZERO:
            # Ray is parallel to plane
            return []
        
        t = -np.dot(origin, self._normal) / denominator
        
        if t < EPS_ZERO:
            # Intersection is behind ray origin or at origin
            return []
        
        intersection_point = origin + t * direction
        
        # Check if intersection is within plane bounds
        if self.is_on_surface(tuple(intersection_point)):
            return [tuple(intersection_point)]
        else:
            return []

    def normal(self, surface_point):
        """ Return the plane normal (same everywhere).
        
        Parameters
        ----------
        surface_point : tuple
            Point on the surface (not used for planes, but kept for consistency)
            
        Returns
        -------
        tuple
            The outward normal vector
        """
        if not self.is_on_surface(surface_point):
            raise GeometryError(
                "Point is not on surface.", 
                {"point": surface_point, "geometry": self}
            )
        return tuple(self._normal)

    def is_entering(self, surface_point, direction):
        """ Returns True if the ray is heading into the plane (negative side).
        
        Parameters
        ---------- 
        surface_point : tuple
            Point on the plane surface
        direction : tuple
            Ray direction vector
            
        Returns
        -------
        bool
            True if ray is entering the negative side of the plane
        """
        if not self.is_on_surface(surface_point):
            raise GeometryError(
                "Point is not on surface.", 
                {"point": surface_point, "geometry": self}
            )
        
        # Ray is entering if it's going in the opposite direction to the normal
        return np.dot(self._normal, direction) < 0.0