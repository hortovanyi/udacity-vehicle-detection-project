import numpy as np
import cv2
import abc
from skimage.feature import hog


class ImageFeaturesBase(abc.ABC):
    @property
    @abc.abstractmethod
    def image(self):
        pass

    @property
    @abc.abstractmethod
    def features(self):
        pass


class Features(object):
    """Features Class - value holder and convenience functions"""
    def __init__(self, values):
        self.__values = np.array(values)

    @property
    def values(self):
        return np.copy(self.__values).astype(np.float64)

    @property
    def features(self):
        return self

    def __add__(self, other):
        if isinstance(other, ImageFeaturesBase):
            other_values = other.features.values
        elif isinstance(other, np.ndarray):
            other_values = other
        else:
            other_values = other.values

        return Features(np.concatenate((self.values, other_values)))

    def __str__(self):
        return str(self.__values)

    def __repr__(self):
        return repr(self.__values)


class HogImageFeatures(ImageFeaturesBase):
    """HogImageFeatures Class"""
    def __init__(self, image, orient, pix_per_cell, cell_per_block,
                 transform_sqrt=False):

        self.__image = image
        self.__orient = orient
        self.__pix_per_cell = pix_per_cell
        self.__cell_per_block = cell_per_block
        self.__transform_sqrt = transform_sqrt
        self.__hog_features = hog(image, orientations=orient,
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block,
                                                   cell_per_block),
                                  transform_sqrt=transform_sqrt,
                                  visualise=False, feature_vector=False)

    @property
    def image(self):
        return self.__image

    @property
    def features(self):
        # vectorise the returned - storing it once
        return Features(self.__hog_features.ravel())

    def window_hog_features(self, window_box):

        pix_per_cell = self.__pix_per_cell
        cell_per_block = self.__cell_per_block
        orient = self.__orient
        # note will only ever be for one channel
        hog_features = self.__hog_features
        # print("hog_features.shape: ", hog_features.shape)
        # copied from Ryans Vehcle Detection Walkthrough
        # nxblocks = (hog_features.shape[1] // pix_per_cell) - 1
        # nyblocks = (hog_features.shape[0] // pix_per_cell) - 1
        # nfeat_per_block = orient*cell_per_block**2
        window = 64
        nblocks_per_window = (window // pix_per_cell) - 1

        # unpack window coordinates and convert to cells
        wa = np.array(window_box)
        # print("window ",window)
        # print("wa ", repr(wa))
        (x1, x2) = wa[:, 0] // pix_per_cell
        (y1, y2) = wa[:, 1] // pix_per_cell

        # print("(x1,x2) =", (x1,x2))
        # print("(y1,y2) =", (y1,y2))
        # print("nblocks_per_window", nblocks_per_window)

        hog_window = hog_features[y1:y1+nblocks_per_window,
                                  x1:x1+nblocks_per_window]

        return Features(hog_window.ravel())

    def window_hog_values(self, window):
        return self.window_hog_features(window).values

    @property
    def values(self):
        return self.features.values

    def __add__(self, other):
        if isinstance(other, np.ndarray):
            other_values = other
        else:
            other_values = other.values
        return np.concatenate((self.values, other_values))

    @property
    def hog_image(self):
        image = self.__image
        orient = self.__orient
        pix_per_cell = self.__pix_per_cell
        cell_per_block = self.__cell_per_block
        transform_sqrt = self.__transform_sqrt

        _, hog_image = hog(image, orientations=orient,
                           pixels_per_cell=(pix_per_cell,
                                            pix_per_cell),
                           cells_per_block=(cell_per_block,
                                            cell_per_block),
                           transform_sqrt=transform_sqrt,
                           visualise=True, feature_vector=False)
        return hog_image


ImageFeaturesBase.register(HogImageFeatures)


class ChannelHistFeatures(ImageFeaturesBase):
    def __init__(self, image, channel=0, nbins=32, bins_range=(0, 256)):
        if image.shape[2] > 1:
            self.__image = image[:, :, channel]
        else:
            self.__image = image
        self.__nbins = nbins
        self.__bins_range = bins_range

        assert len(self.__image.shape) == 2, "Works on one color channel only"

    @property
    def nbins(self):
        return self.__nbins

    @property
    def image(self):
        return self.__image

    @property
    def color_channels(self):
        return 1

    @property
    def features(self):
        return Features(self.channel_hist(self.__image, self.__nbins,
                                          self.__bins_range))

    @property
    def values(self):
        return self.features.values

    @staticmethod
    def channel_hist(img, nbins=32, bins_range=(0, 255)):
        return np.histogram(img, bins=nbins, range=bins_range)[0]

    def __add__(self, other):
        if isinstance(other, np.ndarray):
            other_values = other
        else:
            other_values = other.values
        return np.concatenate((self.values, other_values))


class ColorHistFeatures(ImageFeaturesBase):
    def __init__(self, image, nbins=32, bins_range=(0, 256)):

        self.__image = image
        self.__nbins = nbins
        self.__bins_range = bins_range
        self.__hist_features = None

        # derive the features
        self.__hist_features = Features([])
        channel_values = []
        for channel in range(self.color_channels):
            hist = ChannelHistFeatures.channel_hist(
                self.__image[:, :, channel], self.__nbins, self.__bins_range)
            channel_values.append(hist)

        self.__hist_features = Features(np.concatenate(channel_values))

    @property
    def image(self):
        return self.__image

    @property
    def color_channels(self):
        return self.__image.shape[2]

    @property
    def nbins(self):
        return self.__nbins

    @property
    def features(self):
        return self.__hist_features

    @property
    def values(self):
        return self.features.values

    @staticmethod
    def color_hist(img, nbins=32, bins_range=(0, 256)):
        # Udacity course example

        # Compute the histogram of the channels separately
        hist1 = np.histogram(img[:, :, 0], bins=nbins, range=bins_range)
        hist2 = np.histogram(img[:, :, 1], bins=nbins, range=bins_range)
        hist3 = np.histogram(img[:, :, 2], bins=nbins, range=bins_range)
        # Generating bin centers
        bin_edges = hist1[1]
        bin_centers = (bin_edges[1:] + bin_edges[0:len(bin_edges)-1])/2
        # Concatenate the histograms into a single feature vector
        hist_features = np.concatenate((hist1[0], hist2[0], hist3[0]))
        # Return the individual histograms, bin_centers and feature vector
        return hist1, hist2, hist3, bin_centers, hist_features

    @staticmethod
    def color_hist_features(img, nbins=32, bins_range=(0, 256)):
        # this class was too compute intensive when processing images
        # so created this helper method
        # Compute the histogram of the channels separately
        hist1 = np.histogram(img[:, :, 0], bins=nbins, range=bins_range)
        hist2 = np.histogram(img[:, :, 1], bins=nbins, range=bins_range)
        hist3 = np.histogram(img[:, :, 2], bins=nbins, range=bins_range)

        # Concatenate the histograms into a single feature vector
        hist_features = np.concatenate((hist1[0], hist2[0], hist3[0]))
        # Return the individual histograms, bin_centers and feature vector
        return hist_features

    def __add__(self, other):
        if isinstance(other, np.ndarray):
            other_values = other
        else:
            other_values = other.values
        return np.concatenate((self.values, other_values))


ImageFeaturesBase.register(ChannelHistFeatures)
ImageFeaturesBase.register(ColorHistFeatures)


class BinSpatialFeatures(ImageFeaturesBase):
    def __init__(self, image, color_space='BGR', size=(32, 32)):

        self.__image = image
        self.__size = size
        self.__color_space = color_space

    @property
    def image(self):
        return self.__image

    @property
    def size(self):
        return self.__size

    @property
    def color_space(self):
        return self.__color_space

    @property
    def features(self):
        return Features(self.bin_spatial(self.image,
                                         self.color_space, self.size))

    @property
    def values(self):
        return self.features.values

    def __add__(self, other):
        if isinstance(other, np.ndarray):
            other_values = other
        else:
            other_values = other.values
        return np.concatenate((self.values, other_values))

    @staticmethod
    def bin_spatial(img, color_space='BGR', size=(32, 32)):
        # Convert image to new color space (if specified)
        if color_space != 'BGR':
            if color_space == 'HSV':
                feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            elif color_space == 'LUV':
                feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2LUV)
            elif color_space == 'HLS':
                feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
            elif color_space == 'YUV':
                feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
            elif color_space == 'YCrCb':
                feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        else:
            feature_image = np.copy(img)
        # Use cv2.resize().ravel() to create the feature vector
        features = cv2.resize(feature_image, size).ravel()
        # Return the feature vector
        return features


ImageFeaturesBase.register(BinSpatialFeatures)
