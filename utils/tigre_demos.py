import tigre
import numpy as np

def set_geometry(DSD, DSO,
                 nDetector = np.array([512, 512]),
                 dDetector = np.array([0.8, 0.8]),
                 nVoxel = np.array([256, 256, 256]),
                 dVoxel = np.array([1, 1, 1]),
                 offOrigin = np.array([0, 0, 0]),
                 offDetector = np.array([0, 0]),
                 accuracy = 0.5,
                 COR = 0,
                 rotDetector = np.array([0, 0, 0]),
                 mode = "cone",
                 angles = np.linspace(0, 2 * np.pi, 2400, dtype=np.float32)):

    geo = tigre.geometry(mode="cone", high_resolution=True)

    geo.angles = angles
    # VARIABLE                                   DESCRIPTION                    UNITS
    # -------------------------------------------------------------------------------------
    # Distances
    geo.DSD = DSD  # Distance Source Detector      (mm)
    geo.DSO = DSO  # Distance Source Origin        (mm)
    # Detector parameters
    geo.nDetector = np.array(nDetector)  # number of pixels              (px)
    geo.dDetector = np.array(dDetector)  # size of each pixel            (mm)
    geo.sDetector = np.array(nDetector) * np.array(dDetector)  # total size of the detector    (mm)
    # Image parameters
    geo.nVoxel = np.array(nVoxel)  # number of voxels              (vx)
    geo.dVoxel = np.array(dVoxel)   # size of each voxel            (mm)
    geo.sVoxel = np.array(nVoxel) * np.array(dVoxel)  # total size of the image       (mm)
    # Offsets
    geo.offOrigin = offOrigin  # Offset of image from origin   (mm)
    geo.offDetector = np.array(offDetector)   # Offset of Detector            (mm)
    # These two can be also defined
    # per angle

    # Auxiliary
    geo.accuracy = accuracy  # Variable to define accuracy of
    # 'interpolated' projection
    # It defines the amoutn of
    # samples per voxel.
    # Recommended <=0.5             (vx/sample)

    # Optional Parameters
    # There is no need to define these unless you actually need them in your
    # reconstruction


    geo.COR = COR  # y direction displacement for
    # centre of rotation
    # correction                   (mm)
    # This can also be defined per
    # angle

    geo.rotDetector = rotDetector  # Rotation of the detector, by
    # X,Y and Z axis respectively. (rad)
    # This can also be defined per
    # angle

    geo.mode = mode  # Or 'parallel'. Geometry type.

    return geo