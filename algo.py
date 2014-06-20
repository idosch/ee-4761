import numpy as np
import pymorph
import pylab as plt
from scipy import ndimage


class HEID:
    def __init__(self, frame, sigma_f, r, min_var, r_med):
        self.frame = frame
        self.sigma_f = sigma_f
        self.r = r
        self.min_var = min_var
        self.r_med = r_med

    def start():
        pass


class KTH:
    def __init__(self, image, sigma_s, sigma_b, alpha, tau,
                 watershed=None, sigma_w=None, h_min=None, a_min=None,
                 s_min=None, image_gt=None):
        """Initialize segmentation process of 'frame'.
        1. 'image': original image.
        2. 'sigma_s': variance of Gaussian used to emphasize cells.
        3. 'sigma_b': variance of Gaussian used to emphasize background.
        4. 'alpha': heuristically determined paramter used to subtract the
                    background image from the foreground image.
        5. 'tau': threshold.
        6. watershed: whether to use watershed transform on the distance
                      transform of the segmentation mask or not.
        7. 'sigma_w': variance of Gaussian used for smoothing.
        8. 'h_min': H-minima transform parameter.
        9. 'a_min': minimum area of each cell.
        10. 's_min': minimum summed intensity of each cell.
        11. 'image_gt': corresponding ground truth segmented image from ../TRA/
                        folder.
        """
        self._image = image
        self._image_gt = image_gt
        self._sigma_s = sigma_s
        self._sigma_b = sigma_b
        self._alpha = alpha
        self._tau = tau
        self._watershed = watershed
        self._sigma_w = sigma_w
        self._h_min = h_min
        self._a_min = a_min
        self._s_min = s_min
        self._image_gt = image_gt

    def start(self):
        """Segment frame.

        The returned value is a labeled uint16 image.
        """
        # Preprocessing: subtract minimum pixel value.
        I = self._image - self._image.min()
        # 'Bandpass' filtering.
        I_s = ndimage.filters.gaussian_filter(I, self._sigma_s)  # Foreground.
        I_b = ndimage.filters.gaussian_filter(I, self._sigma_b)  # Background.
        I_bp = I_s - self._alpha * I_b
        # Thresholding: create binary image.
        I_bin = (I_bp > self._tau)
        # Hole filling.
        I_bin = ndimage.binary_fill_holes(I_bin > 0)

        I_cells = ndimage.label(I_bin)[0]
        # Avoid merging nearby cells using watershed.
        if self._watershed:
            # Distance transfrom on which to apply the watershed algorithm.
            I_dist = ndimage.distance_transform_edt(I_bin)
            I_dist = I_dist/float(I_dist.max()) * 255
            I_dist = I_dist.astype(np.uint8)
            # Find markers for the watershed algorithm.
            # Reduce false positive using Gaussian smoothing.
            I_mask = ndimage.filters.gaussian_filter(I_dist, 8)*I_bin
            rmax = pymorph.regmax(I_mask)
            I_markers, num_markers = ndimage.label(rmax)
            I_dist = I_dist.max() - I_dist  # Cells are now the basins.

            I_cells = pymorph.cwatershed(I_dist, I_markers)

        # Remove cells with area less than threshold.
        if self._a_min:
            for label in np.unique(I_cells)[1:]:
                if (I_cells == label).sum() < self._a_min:
                    I_cells[I_cells == label] = 0

        # Remove cells with summed intensity less than threshold.
        if self._s_min:
            for label in np.nditer(np.unique(I_cells)[1:]):
                if I_bp[I_cells == label].sum() < self._s_min:
                    I_cells[I_cells == label] = 0

        return I_cells.astype('uint16')  # This data type is used by ISBI.
