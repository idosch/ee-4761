import numpy as np
import mahotas as mh
import pymorph
from scipy import ndimage
from skimage import measure
from skimage import morphology as morph
import pylab as plt


class HEID:
    def __init__(self, frame, sigma_f, r, min_var, r_med, a_min, r1=None,
                 r2=None, debug=None):
        self._frame = frame
        self._sigma_f = sigma_f
        self._r = r
        self._min_var = min_var
        self._r_med = r_med
        self._r1 = r1
        self._r2 = r2
        self._a_min = a_min
        self._debug = debug

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
        I_mask = self._segment(I, True).astype('bool')
        mean_fore = I[I_mask].mean()
        I[I_mask] = mean_fore
        I = ndimage.filters.gaussian_filter(I, self._sigma_f)
        I_label = self._segment(I.copy(), False)

        return I_label

    def _segment(self, I, first):
        """Return the segmented frame 'I'.

        If 'first is True, then this is the first segmentation iteration,
        otherwise the second.

        The returned value is a labeled image of type uint16, in order to be
        compatible with ISBI's tool.
        """
        # Compute global threshold.
        otsu_thresh = mh.thresholding.otsu(I.astype('uint16'))
        # Threshold using global and local thresholds.
        fnc = fnc_class(I.shape)
        I_bin = ndimage.filters.generic_filter(I, fnc.filter, size=self._r,
                                               extra_arguments=(I, self._min_var,
                                                                otsu_thresh))

        I_med = ndimage.filters.median_filter(I_bin, size=self._r_med)
        # Remove cells which are too small (leftovers).
        labeled = mh.label(I_med)[0]
        sizes = mh.labeled.labeled_size(labeled)
        too_small = np.where(sizes < self._a_min)
        I_cleanup = mh.labeled.remove_regions(labeled, too_small)
        I_cleanup = mh.labeled.relabel(I_cleanup)[0]

        # Fill holes.
        I_holes = ndimage.morphology.binary_fill_holes(I_cleanup > 0)

        # Binary closing.
        if first and self._r1:
            # First iteration.
            I_morph = morph.binary_closing(I_holes, morph.disk(self._r1))
        elif not first and self._r2:
            # Second iteration.
            I_morph = morph.binary_closing(I_holes, morph.disk(self._r2))
        else:
            # No binary closing.
            I_morph = I_holes

        # Fill yet to be filled holes.
        labels = measure.label(I_morph)
        labelCount = np.bincount(labels.ravel())
        background = np.argmax(labelCount)
        I_morph[labels != background] = True

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

        if self._debug:
            plt.subplot(2, 4, 1)
            plt.imshow(I)
            plt.title('Original Image')
            plt.subplot(2, 4, 2)
            plt.imshow(I_bin)
            plt.title('After Thresholding')
            plt.subplot(2, 4, 3)
            plt.imshow(I_med)
            plt.title('After Median Filter')
            plt.subplot(2, 4, 4)
            plt.imshow(I_cleanup)
            plt.title('After Cleanup')
            plt.subplot(2, 4, 5)
            plt.imshow(I_holes)
            plt.title('After Hole Filling')
            plt.subplot(2, 4, 6)
            plt.imshow(I_morph)
            plt.title('After Closing')
            plt.subplot(2, 4, 7)
            plt.imshow(I_label)
            plt.title('Labeled Image')
            plt.show()

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
        else:
            thresh = glob_thresh

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

        Note: this function takes a long time to complete, thus, we decided to
        use the mean value in each window as a threshold instead of minimum
        error thresholding.
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
                mean1, sigma1 = histogram1.mean(), histogram1.std()
                mean2, sigma2 = histogram2.mean(), histogram2.std()

                # Compute the criterion function only if both classes contain
                # at least two intensity values.
                if sigma1 > 0 and sigma2 > 0:
                    J[T-1] = (1 + 2 * (P1 * np.log(sigma1) + P2 * np.log(sigma2))
                              - 2 * (P1 * np.log(P1) + P2 * np.log(P2)))

        # Find the value of T, which minimizes J.
        return np.argmin(J)


class ilastik:
    def __init__(self, frame, a_min=None, fill=None):
        self._frame = frame
        self._a_min = a_min
        self._fill = fill

    def start(self):
        """Segment the frame.

        The returned value is a labeled uint16 image.
        """
        background = np.bincount(self._frame.ravel()).argmax()  # Most common value.
        I_label = measure.label(self._frame, background=background)
        I_label += 1  # Background is labeled as -1, make it 0.
        I_bin = I_label > 0

        # Remove cells which are too small (leftovers).
        if self._a_min:
            I_label = mh.label(I_bin)[0]
            sizes = mh.labeled.labeled_size(I_label)
            too_small = np.where(sizes < self._a_min)
            I_cleanup = mh.labeled.remove_regions(I_label, too_small)
            I_bin = I_cleanup > 0

        # Fill holes.
        if self._fill:
            I_bin = ndimage.morphology.binary_fill_holes(I_bin)  # Small holes.
            # Bigger holes.
            labels = measure.label(I_bin)
            label_count = np.bincount(labels.ravel())
            background = np.argmax(label_count)
            I_bin[labels != background] = True

        I_label = mh.label(I_bin)[0].astype('uint16')
        return I_label


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
