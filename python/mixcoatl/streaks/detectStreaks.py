import cv2
import numpy as np
from skimage.feature import hessian_matrix, hessian_matrix_eigvals

from lsst.geom import Point2I
from lsst.pex.config import Config, Field
from lsst.pipe.base import Struct, Task
from mixcoatl.streaks.line import fit_line_from_xy

class HessianDetectConfig(Config):
    """Configurable parameters for StreakFinderTask.
    """
    bin_size = Field(
        doc="Size of superpixel bins.",
        dtype=int,
        default=4
    )
    kernel = Field(
        doc="Size of Gaussian kernel for initial smoothing.",
        dtype=int,
        default=11,
    )
    sigma = Field(
        doc="Standard deviation of the Gaussian kernel used to compute the Hessian second derivatives.",
        dtype=float,
        default=12.0,
    )
    edge = Field(
        doc="Number of edge pixels to zero out in the binned image before thresholding.",
        dtype=int,
        default=20,
    )
    threshold = Field(
        doc="Threshold applied to the Hessian minima ridges eigenvalue array.",
        dtype=float,
        default=-0.05,
    )
    aspect = Field(
        doc="Lower bound for the aspect ratio of regions, after thresholding",
        dtype=float,
        default=8.0,
    )
    limit = Field(
        doc="Limit of the deviation from horizontal/vertical orientation for streak exclusion",
        dtype=float,
        default=0.10,
    )


class HessianDetectTask(Task):
    ConfigClass = HessianDetectConfig
    _DefaultName = "hessianDetect"

    def run(self, exposure):

        arr = exposure.getImage().getArray()

        # Bin original image down to binxbin pixels
        bin_size = self.config.bin_size
        arr = np.clip(arr, a_min=0, a_max=100)
        new_shape = (int(arr.shape[0] / bin_size), int(arr.shape[1] / bin_size))
        
        # Rebin by averaging
        bin_arr = arr.reshape(
            new_shape[0],
            arr.shape[0] // new_shape[0],
            new_shape[1],
            arr.shape[1] // new_shape[1]
        ).mean(-1).mean(1)
    
        # Use the Hessian matrix to find streaks
        # The minima ridges output has been most effective
        # in finding the streaks
        kernel = self.config.kernel
        gauss = cv2.GaussianBlur(bin_arr, (kernel, kernel), 0) # Blur with Gaussian kernel

        sigma = self.config.sigma
        h_elems = hessian_matrix(gauss, sigma=sigma, order='rc', use_gaussian_derivatives=False)
        maxima_ridges, minima_ridges = hessian_matrix_eigvals(h_elems)
        
        # Now we create a binary image 
        # Setting this threshold has been tricky
        threshold = self.config.threshold
        binary_ridges = minima_ridges < threshold
        binary_ridges = binary_ridges.astype(np.uint8)
        
        # Set edges of binary_ridges to zero
        edge = self.config.edge
        binary_ridges[:,0:edge] = 0
        binary_ridges[:,-edge:-1] = 0
        binary_ridges[0:edge,:] = 0
        binary_ridges[-edge:-1,:] = 0
        
        # Convert to 0 -> 255
        _, binary = cv2.threshold(binary_ridges, 0.5, 255, cv2.THRESH_BINARY)
        
        # Find connected regions
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        
        # Sort to find regions with long aspect ratios
        long_labels = []
        aspect = self.config.aspect
        for i in range(num_labels):
            mask = np.uint8(labels == i)
            # Extract points (x,y) of this component
            ys, xs = np.where(mask > 0)
            points = np.column_stack((xs, ys))
            rect = cv2.minAreaRect(points)
            (center, (width, height), angle) = rect
            if height > 0 and width > 0:
                aspect_ratio = max(width, height) / min(width, height)
            else:
                aspect_ratio = 0  # Handle division by zero for flat regions
            if aspect_ratio > aspect:
                long_labels.append(i)
    
        r, c = arr.shape # Debugging code: swapped c, r to r, c
        # Fit lines to the longest ones
        lines = []

        for label in long_labels:
            mask = np.uint8(labels == label)
            # Extract points (x,y) of this component
            ys, xs = np.where(mask > 0)

            line = fit_line_from_xy(xs, ys)
            line.rescale(float(bin_size))
            lines.append(line)

        return Struct(
            lines=lines,
            minima_ridges=minima_ridges,
            binary_ridges=binary_ridges,
        )
