from skimage.feature import hessian_matrix, hessian_matrix_eigvals
import cv2
import numpy as np

import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import lsst.afw.image as afwImage
import lsst.pipe.base.connectionTypes as cT
from lsst.meas.algorithms import MaskStreaksTask
from lsst.meas.algorithms.maskStreaks import LineCollection
from lsst.geom import Point2I

class StreakFinderConfig(pexConfig.Config):
    """Configurable parameters for StreakFinderTask.
    """
    binsize = pexConfig.Field(
        dtype=int,
        default=4,
        doc="Size of pixel binning of the image.",
    )

    kernel = pexConfig.Field(
        dtype=int,
        default=11,
        doc="Size in pixels of Gaussian kernel for initial smoothing.",
    )

    sigma = pexConfig.Field(
        dtype=float,
        default=12.0,
        doc="Standard deviation of the Gaussian kernel used to compute the Hessian second derivatives.",
    )

    edge = pexConfig.Field(
        dtype=int,
        default=20,
        doc="Number of edge pixels to zero out in the binned image before thresholding.",
    )

    threshold = pexConfig.Field(
        dtype=float,
        default=-0.05,
        doc="Threshold applied to the Hessian minima ridges eigenvalue array.",
    )

    aspect = pexConfig.Field(
        dtype=float,
        default=8.0,
        doc="Lower bound for the aspect ratio of regions, after thresholding",
    )

    streakWidth = pexConfig.Field(
        dtype=int,
        default=100,
        doc="Width of the streak which is drawn on the image",
    )

    limit = pexConfig.Field(
        dtype=float,
        default=0.10,
        doc="Limit of the deviation from horizontal/vertical orientation for streak exclusion",
    )

class StreakFinderTask(pipeBase.Task):
    ConfigClass = StreakFinderConfig
    _DefaultName = "streakFinder"

    @timeMethod
    def run(self, exposure):

        arr = exposure.getImage().getArray()    

        # Bin original image down to binxbin pixels
        binsize = self.config.binsize
        arr = np.clip(arr, a_min=0, a_max=100)
        new_shape = (int(arr.shape[0] / binsize), int(arr.shape[1] / binsize))
        
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
        H_elems = hessian_matrix(gauss, sigma=sigma, order='rc', use_gaussian_derivatives=False)
        maxima_ridges, minima_ridges = hessian_matrix_eigvals(H_elems)
        
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
        num_labels, labels, stats, centroids = \
            cv2.connectedComponentsWithStats(binary, connectivity=8)
        
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
            if height > 0:
                aspect_ratio = max(width, height) / min(width, height)
            else:
                aspect_ratio = 0  # Handle division by zero for flat regions
            if aspect_ratio > aspect:
                long_labels.append(i)
    
        r, c = arr.shape # Debugging code: swapped c, r to r, c
        # Fit lines to the longest ones
        rhos = []
        thetas = []
        lines = LineCollection([], [])

        test_lines = [] ## Debugging code

        for label in long_labels:
            mask = np.uint8(labels == label)
            # Extract points (x,y) of this component
            ys, xs = np.where(mask > 0)
            points = np.column_stack((xs, ys))
            
            # Fit a line through the points
            # x0, y0 are shape centroid
            # vx, vy are a normlized vector in the direction of the line
            # Resize x0 and y0 to the original image
            [vx, vy, x0, y0] = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01)
            vx = vx[0]; vy = vy[0]; x0 = x0[0] * binsize; y0 = y0[0] * binsize
            
            # Weed out near horizontal or vertical lines
            limit = self.config.limit
            if (abs(vx) < limit) or (abs(vy) < limit):
                continue

            ## Debugging code
            # Find points at the edges
            alpha = min((r - 1 - x0) / vx, (c - 1 - y0) / vy)
            right_point = (int(x0 + alpha * vx), int(y0 + alpha * vy))
            beta = min(x0 / vx , y0 / vy)
            left_point = (int(x0 - beta * vx), int(y0 - beta * vy))
            test_lines.append([Point2I(left_point), Point2I(right_point)])
            print(left_point, right_point)
                
            # Now find rho, theta for lsst.meas.algorithms.maskStreaks.Line class
            theta = -np.atan2(vx, vy)
            rho = (x0 - c/2) * np.cos(theta) + (y0 - r/2) * np.sin(theta)
            theta *= 180.0 / np.pi # Convert to degrees
            if theta < 0.:
                rho *= -1
                theta += 180.
            rhos.append(rho)
            thetas.append(theta)
            lines = LineCollection(np.array(rhos), np.array(thetas))

        return pipeBase.Struct(
            lines=lines,
            minima_ridges=minima_ridges,
            binary_ridges=binary_ridges,
        )

class DetectStreaksTaskConnections(pipeBase.PipelineTaskConnections,
                                   dimensions=("instrument", "visit", "detector")):

    exposure = connectionTypes.Input(
        doc="Background-subtracted exposure to detect streaks on.",
        name="preliminary_visit_image",
        storageClass="Exposure",
        dimensions=("instrument", "visit", "detector"),
    )
    detectedLines = connectionTypes.Output(
        doc="Lines detected in the input exposure.",
        name="detected_lines",
        storageClass="StructuredDataDict",
        dimensions=("instrument", "visit", "detector"),
    )

    # kht exclusive outputs
    originalLines = connectionTypes.Output(
        doc="Lines identified by kernel hough transform.",
        name="original_lines",
        storageClass="StructuredDataDict",
        dimensions=("instrument", "visit", "detector"),
    )

    # hessian exclusive outputs
    minimaRidges = connectionTypes.Output(
        doc="Hessian matrix minima ridges image.",
        name="minima_ridges",
        storageClass="Image",
        dimensions=("instrument", "visit", "detector"),
    )
    binaryRidges = connectionTypes.Output(
        doc="Detected ridges image.",
        name="binary_ridges",
        storageClass="Mask",
        dimensions=("instrument", "visit", "detector"),
    )

    def __init__(self, *, config=None):
        super().__init__(config=config)
        if config.detectionAlgorithm != "hessian":
            del self.minimaRidges
            del self.binaryRidges 
        if config.detectionAlgorithm != "kht":
            del self.originalLines

class DetectStreaksTaskConfig(pipeBase.PipelineTaskConfig,
                              pipelineConnections=DetectStreaksTaskConnections):
    """Configuration parameters for DetectStreaksTask.
    """
    detectionAlgorithm = pexConfig.ChoiceField(
        dtype=str,
        default="kht",
        doc="Line detection algorithm to use.",
        allowed={
            "kht" : "Kernel Hough transform.",
            "hessian" : "Hessian matrix.",
        }
    )
    maskStreaks = pexConfig.ConfigurableField(
        target=MaskStreaksTask,
        doc="Detect streaks using Kernel Hough transform."
    )
    streakFinder = pexConfig.ConfigurableField(
        target=StreakFinderTask,
        doc="Detect streaks using Hessian matrix line detection."
    )


class DetectStreaksTask(pipeBase.PipelineTask):
    """Find streaks or other straight lines in the image.
    """

    ConfigClass = DetectStreaksTaskConfig
    _DefaultName = "detectStreaks"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.makeSubtask('maskStreaks')
        self.makeSubtask('streakFinder')

    def run(self, exposure):
        """Find streaks in the image.

        Find streaks in a background subtracted image using either a Kernal 
        Hough Transform or a Hessian matrix line detection.

        Parameters
        ----------
        exposure : `lsst.afw.image.Exposure`
            A background subtracted exposure.

        Returns
        -------
        result : `lsst.pipe.base.Struct`
            Results as a struct with attributes:

            ``lines``
                Final results for lines. (`dict`)
            ``originalLines``
                Lines identified by the kernel hough transform. (`dict`)
            ``minimaRidges``
                The image of minima ridges output by the Hessian matrix. 
                (`lsst.afw.image.Image`)
            ``binaryRidges``
                The mask of minima ridges below the detection threshold. 
                (`lsst.afw.image.Mask`)
        """
        result = pipeBase.Struct(
            detectedLines=None,
            originalLines=None,
            minimaRidges=None,
            binaryRidges=None,
        )
        if self.config.detectionAlgorithm == 'kht':
            detectedLineResults = self.maskStreaks.run(exposure.getMaskedImage())
            result.originalLines = detectedLineResults.originalLines
        elif self.config.detectionAlgorithm == 'hessian':
            detectedLineResults = self.streakFinder.run(exposure)
            result.minimaRidges = afwImage.ImageF(detectedLineResults.minima_ridges.astype(np.float32))
            result.binaryRidges = afwImage.MaskX(detectedLineResults.binary_ridges.astype(np.int32))

        lines = detectedLineResults.lines
        linesDict = {'rhos' : lines.rhos.tolist(),
                     'thetas' : lines.thetas.tolist(),
                     'sigmas' : lines.sigmas.tolist(),
                    }
        result.detectedLines = linesDict

        return result
