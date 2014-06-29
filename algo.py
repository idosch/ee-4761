import numpy as np
import pymorph
import mahotas as mh
import pylab as plt
from scipy import ndimage
from skimage import morphology as morph
from skimage import measure
import pdb


class HEID:
    def __init__(self, frame, sigma_f, r, min_var, r_med):
        self._frame = frame
        self._sigma_f = sigma_f
        self._r = r
        self._min_var = min_var
        self._r_med = r_med

    def start(self):
        """Segment the frame.

        The returned value is a labeled uint16 image.
        """
        # Preprocessing.
        I = self._frame / np.float(self._frame.max()) * 255.0
        I = ndimage.filters.gaussian_filter(I, self._sigma_f)

        # Run the algorithm for the first time to produce a mask. Then, use
        # this mask the replace the bright pixels with the mean value and
        # rerun the algorithm.
        #I_label = self._segment(I).astype('bool')
        I_mask = self._segment(I, True).astype('bool')
        mean_fore = I[I_mask].mean()
        I[I_mask] = mean_fore
        I = ndimage.filters.gaussian_filter(I, self._sigma_f)
        I_label = self._segment(I.copy(), False)

        return I_label

    def _segment(self, I, first):
        # Thresholding.
        otsu_thresh = mh.thresholding.otsu(I.astype('uint16'))
        fnc = fnc_class(I.shape)
        I_bin = ndimage.filters.generic_filter(I, fnc.filter, size=self._r,
                                               extra_arguments=(I, self._min_var,
                                                                otsu_thresh))
        # Remove cells which are too small (leftovers).
        I_morph = ndimage.filters.median_filter(I_bin, size=self._r_med)
        I_label = mh.label(I_morph)[0]
        sizes = mh.labeled.labeled_size(I_label)
        too_small = np.where(sizes < 100)
        I_label = mh.labeled.remove_regions(I_label, too_small)
        I_label = mh.labeled.relabel(I_label)[0]
        # Fill holes.
        I_morph = ndimage.morphology.binary_fill_holes(I_label > 0)
        I_morph0 = I_morph.copy()
        if first:
            I_morph = morph.binary_closing(I_morph, morph.disk(3))
        else:
            I_morph = morph.binary_closing(I_morph, morph.disk(6))
        labels = measure.label(I_morph)
        labelCount = np.bincount(labels.ravel())
        background = np.argmax(labelCount)
        I_morph[labels != background] = True

        I_morph1 = I_morph.copy()
        I_morph = I_morph.astype('uint16')

        # Separate touching cells using watershed.
        # Distance transfrom on which to apply the watershed algorithm.
        I_dist = ndimage.distance_transform_edt(I_morph)
        I_dist = I_dist/float(I_dist.max()) * 255
        I_dist = I_dist.astype(np.uint8)
        # Find markers for the watershed algorithm.
        # Reduce false positive using Gaussian smoothing.
        I_mask = ndimage.filters.gaussian_filter(I_dist, 8)*I_morph
        rmax = pymorph.regmax(I_mask)
        I_markers, _ = ndimage.label(rmax)
        I_dist = I_dist.max() - I_dist  # Cells are now the basins.
        I_label = pymorph.cwatershed(I_dist, I_markers)

        """
        if not first:
            plt.subplot(2, 3, 1)
            plt.title('Original Image')
            plt.imshow(self._frame)
            plt.subplot(2, 3, 2)
            plt.title('After Smoothing')
            plt.imshow(I)
            plt.subplot(2, 3, 3)
            plt.title('After Thresholding')
            plt.imshow(I_bin, cmap=plt.cm.gray)
            plt.subplot(2, 3, 4)
            plt.title('After Morphological Operations')
            plt.imshow(I_morph0, cmap=plt.cm.gray)
            plt.subplot(2, 3, 5)
            plt.title('After Hole Filling')
            plt.imshow(I_morph1, cmap=plt.cm.gray)
            plt.subplot(2, 3, 6)
            plt.title('After Watershed Transform')
            plt.imshow(I_label)
            plt.show()
        """
        return I_label.astype('uint16')


class fnc_class:
    def __init__(self, shape):
        # store the shape:
        self.shape = shape
        # initialize the coordinates (row, col):
        self.coordinates = [0] * len(shape)

    def filter(self, buffer, I, min_var, glob_thresh):
        """Classify a pixel as foreground (True) or background (False).
        
        If the variance of the elements of 'x' is larger than 'min_var', then
        the current considered pixel is thresholded using a local threshold,
        otherwise it is thresholded using the global threshold, 'glob_thresh'.
        """
        if np.var(buffer) > min_var:
            thresh = buffer.mean()
            #thresh = mh.thresholding.otsu(buffer.astype('uint16'))
            #thresh = self.thresh_min_err(buffer)
        else:
            thresh = glob_thresh
        #thresh = glob_thresh

        row, col = self.coordinates[0], self.coordinates[1]
        # calculate the next coordinates:
        axes = range(len(self.shape))
        axes.reverse()
        for jj in axes:
            if self.coordinates[jj] < self.shape[jj] - 1:
                self.coordinates[jj] += 1
                break
            else:
                self.coordinates[jj] = 0

        return I[row, col] > thresh

    def thresh_min_err(self, buffer):
        """Return the threshold for 'buffer'.

        The returned threshold is computed according to J. Kittler &
        J. Illingworth: "Minimum Error Thresholding".

        The code was translated from the following MATLAB implementation:
        http://stackoverflow.com/questions/2055774/adaptive-thresholding
        """
        # Initialize the criterion function.
        J = np.inf * np.ones(255)

        # Compute the probability densitiy function.
        histogram = np.fromiter((np.sum(buffer == x) for x in np.arange(0, 256)),
                                np.int) / float(buffer.size)
        
        # Walk through every possible threshold. However, T is interpreted
        # differently than in the paper. It is interpreted as the lower
        # boundary of the second class of pixels rather than the upper
        # boundary of the first class. That is, an intensity value of T is
        # treated as being in the same class as higher intensities rather
        # than lower intensities.
        for T in np.arange(1, 256):
            # Split the histogram at threshold T.
            histogram1 = histogram[1:T]
            histogram2 = histogram[T:]

            # Compute the probability of each class.
            P1 = histogram1.sum()
            P2 = histogram2.sum()

            # Continue only if both classes aren't empty.
            if P1 > 0 and P2 > 0:
                # Compute the STD of both classes.
                mean1, sigma1  = histogram1.mean(), histogram1.std()
                mean2, sigma2  = histogram2.mean(), histogram2.std()

                # Compute the criterion function only if both classes contain
                # at least two intensity values.
                if sigma1 > 0 and sigma2 > 0:
                    J[T-1] = (1 + 2 * (P1 * np.log(sigma1) + P2 * np.log(sigma2))
                              - 2 * (P1 * np.log(P1) + P2 * np.log(P2)))

        # Find the value of T, which minimizes J.
        return np.argmin(J)


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
