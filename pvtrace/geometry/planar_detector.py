import numpy as np
from pvtrace.geometry.plane import Plane  # Import Plane instead of Box
from pvtrace.material.surface import SurfaceDelegate


class VacuumDetectorDelegate(SurfaceDelegate):
    """ Surface delegate that acts like pure vacuum but detects specific rays.
    """
    
    def __init__(self, detection_direction):
        """
        Parameters
        ----------
        detection_direction : tuple
            Direction from which rays are detected (absorbed)
            For planar detectors, detects entire hemisphere
        """
        self.detection_direction = np.array(detection_direction, dtype=np.float32)
        # Counter for detected rays - stored in the delegate
        self.detected_count = 0
        # Remove ray storage for speed
        self.detected_rays = []
    
    def _is_detection_direction(self, ray_direction):
        """ Check if ray direction is from the detection hemisphere.
        For planar detectors, just check if dot product > 0.
        """
        # Convert to numpy array
        ray_dir = np.array(ray_direction, dtype=np.float32)
        
        # Simple dot product check - no normalization needed!
        # If dot product > 0, ray is coming from the detection hemisphere
        return np.dot(ray_dir, self.detection_direction) > 0
    
    def reflectivity(self, surface, ray, geometry, container, adjacent):
        """ Returns 1.0 (full reflection) for detected rays to stop them.
        """
        if self._is_detection_direction(ray.direction):
            # Just count - don't store ray details for speed
            self.detected_count += 1
            self.detected_rays.append({
                'position': ray.position,
                'direction': ray.direction,
                'wavelength': ray.wavelength
            })
            return 1.0  # Full reflection to "capture" the ray
        else:
            return 0.0  # No reflection - acts like vacuum

    def reflected_direction(self, surface, ray, geometry, container, adjacent):
        """ For detected rays, return zero direction to kill them.
        """
        return (0.0, 0.0, 0.0)  # Kill the ray
            
    def transmitted_direction(self, surface, ray, geometry, container, adjacent):
        """ Rays pass through unchanged for non-detected rays.
        """
        return ray.direction  # Avoid tuple conversion


class PlanarDetector(Plane):  # Inherit from Plane instead of Box
    """ A planar detector that only detects rays approaching from a specific direction.
    """
    
    def __init__(self, length=1.0, width=1.0, normal=(0, 0, 1)):
        """
        Parameters
        ----------
        length : float
            Length of the detector plane
        width : float
            Width of the detector plane
        normal : tuple
            Normal vector of the plane (default: pointing in +z direction)
        """
        # Create a true plane with specified dimensions and normal
        size = (length, width)  # Only need width and height for a plane
        super().__init__(size=size, normal=normal)


def create_planar_detector_node(name, length=1.0, width=1.0, normal=(0, 0, 1), 
                               detection_direction=None, parent=None):
    """ Helper function to create a planar detector node.
    """
    from pvtrace import Node, Material
    from pvtrace.material.surface import Surface
    
    # Create detector geometry using the PlanarDetector class (now a true plane)
    detector_geometry = PlanarDetector(
        length=length, 
        width=width, 
        normal=normal
    )
    
    # Create surface delegate
    if detection_direction is None:
        detection_direction = tuple(-np.array(normal))
    
    detector_delegate = VacuumDetectorDelegate(
        detection_direction=detection_direction
    )
    
    # Create detector node
    detector_node = Node(
        name=name,
        geometry=detector_geometry,
        parent=parent
    )
    
    # Set material with custom surface - same refractive index as container
    if parent and hasattr(parent.geometry, 'material'):
        container_n = parent.geometry.material.refractive_index
    else:
        container_n = 1.0  # Default to vacuum
    
    detector_node.geometry.material = Material(
        refractive_index=container_n,  # Same as container = no refraction
        surface=Surface(delegate=detector_delegate)
    )
    
    # Store reference to delegate for easy access to detection results
    detector_node.detector_delegate = detector_delegate
    
    return detector_node