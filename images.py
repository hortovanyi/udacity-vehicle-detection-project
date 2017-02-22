import cv2
import numpy as np
from boxes import WindowBoxes, WindowBoxSlice
from skimage.feature import hog
from features import Features, BinSpatialFeatures
from features import ColorHistFeatures, HogImageFeatures
from search import SearchParams

from numba import jit


class CameraImage(object):
    """holds an image and does color space conversions """

    def __init__(self, image, color_space='BGR'):
        self.__color_space = color_space

        # want to make sure its stored as BGR if possible
        if color_space == 'BGR':
            self.__image = image
        elif color_space == 'RGB':
            self.__image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            self.__image = image

        # double check scaling
        assert np.max(self.__image) >= 1

        # this gets created as we need it once only
        self.__hog_features = None

    @property
    def image(self):
        return self.__image

    @property
    def height(self):
        return self.__image.shape[0]

    @property
    def width(self):
        return self.__image.shape[1]

    @property
    def channels(self):
        return self.__image.shape[2]

    @property
    def y_center(self):
        return np.int(self.height / 2)

    @property
    def image_bottom(self):
        return self.image[self.y_center:self.height]

    @property
    def rgb(self):
        return cv2.cvtColor(self.__image, cv2.COLOR_BGR2RGB)

    def to_color_space(self, color_space):
        return self.convert_color_space(self.__image, color_space)

    @staticmethod
    def slice_window(image, window):
        return image[window[0][1]:window[1][1], window[0][0]:window[1][0]]

    @staticmethod
    def slice_bounding_box_shape_resize_64x64(image, wb: WindowBoxes,
                                              shape: str):
        """resize to (64,64) to match trained data size"""
        window = wb.bounding_box(shape)
        ratio = wb.resize_64x64_ratio(shape)

        if ratio < 1:
            # shrinking
            image = cv2.resize(
                CameraImage.slice_window(image, window),
                None, fx=ratio, fy=ratio, interpolation=cv2.INTER_AREA)
        else:
            # zooming
            image = cv2.resize(
                CameraImage.slice_window(image, window),
                None, fx=ratio, fy=ratio, interpolation=cv2.INTER_LINEAR)
        return image

    @staticmethod
    def convert_color_space(image, color_space):
        img = np.copy(image)
        if color_space != 'BGR':
            if color_space == 'HSV':
                img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            elif color_space == 'LUV':
                img = cv2.cvtColor(img, cv2.COLOR_BGR2LUV)
            elif color_space == 'HLS':
                img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
            elif color_space == 'YUV':
                img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
            elif color_space == 'YCrCb':
                img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        return img

    def hog_features(self, sp: SearchParams):
        # once off creation of hog_features for the bottom half of the screen
        if self.__hog_features is None:
            hog_channel, orient = sp.hog_channel, sp.orient
            pix_per_cell, cell_per_block = sp.pix_per_cell, sp.cell_per_block
            self.__hog_features = self.hog_features_list(
                self.image_bottom, hog_channel, orient,
                pix_per_cell, cell_per_block)
        return self.__hog_features

    @staticmethod
    def hog_channel_features(img_channel, sp: SearchParams, vis=False,
                             feature_vec=False, transform_sqrt=False):
        orient = sp.orient
        pix_per_cell, cell_per_block = sp.pix_per_cell, sp.cell_per_block
        hif = HogImageFeatures(img_channel, orient,
                               pix_per_cell, cell_per_block)
        return hif.features

    @staticmethod
    def hog_features_list(img, hog_channel, orient, pix_per_cell,
                          cell_per_block, color_space='BGR'):
        if color_space != 'BGR':
            img = CameraImage.convert_color_space(img, color_space)
        hif_list = []
        if hog_channel == 'ALL':
            for channel in range(img.shape[2]):
                hif_list.append(HogImageFeatures(
                    img[:, :, channel], orient, pix_per_cell, cell_per_block))
        else:
            hif_list.append(HogImageFeatures(
                img[:, :, hog_channel], orient, pix_per_cell, cell_per_block))

        return hif_list

    @staticmethod
    def bin_spatial(img, color_space='BGR', size=(32, 32)):
        if color_space != 'BGR':  # if no color_space passed it wont change it
            img = CameraImage.convert_color_space(img, colour_space)
        bsf = BinSpatialFeatures(img, size=spatial_size)
        return bsf.features

    @staticmethod
    def color_hist(img, color_space='BGR', nbins=32, bins_range=(0, 256)):
        if color_space != 'BGR':  # if no color_space passed it wont change it
            img = CameraImage.convert_color_space(img, colour_space)
        chf = ColorHistFeatures(img, nbins=hist_bins, bins_range=bins_range)

        return chf.features


class ImageSlice(object):
    """holds an image slice and does feature searches """

    def __init__(self, camera_image: CameraImage, window_boxes: WindowBoxes,
                 search_params: SearchParams, shape: str):
        self.__camera_image = camera_image
        self.__window_boxes = window_boxes
        self.__search_params = search_params
        self.__shape = shape
        self.__image = None
        self.__image_color_converted = None
        self.__hog_features = None

    @property
    def image(self):
        if self.__image is None:
            camera_image = self.__camera_image
            window_boxes = self.__window_boxes
            shape = self.__shape
            self.__image = CameraImage.slice_bounding_box_shape_resize_64x64(
                camera_image.image, window_boxes, shape)
        return self.__image

    @property
    def image_color_converted(self):
        if self.__image_color_converted is None:
            search_params = self.__search_params
            if search_params.color_space == 'BGR':
                self.__image_color_converted = self.image
            else:
                self.__image_color_converted = CameraImage.convert_color_space(
                    self.image, search_params.color_space)

        return self.__image_color_converted

    @property
    def shape(self):
        return self.__shape

    @property
    def bin_spatial_features(self):
        sp = self.__search_params
        return CameraImage.bin_spatial(self.image_color_converted,
                                       size=sp.spatial_size)

    def slice_window(self, window):
        return self.image_color_converted[window[0][1]:window[1][1],
                                          window[0][0]:window[1][0]]

    def window_bin_spatial_features(self, window):
        sp = self.__search_params
        return CameraImage.bin_spatial(self.slice_window(window),
                                       size=sp.spatial_size)

    @property
    def color_hist_features(self):
        sp = self.__search_params
        return CameraImage.color_hist(self.image_color_converted,
                                      nbins=sp.hist_bins)

    def window_color_hist_features(self, window):
        sp = self.__search_params
#         return CameraImage.color_hist(self.slice_window(window),
#  nbins=search_params.hist_bins,
#                                       bins_range=search_params.hist_range)
        return Features(
            ColorHistFeatures.color_hist_features(self.slice_window(window),
                                                  nbins=sp.hist_bins,
                                                  bins_range=sp.hist_range))

    @property
    def hog_features_list(self):
        # its expensive to create so do it once per image slice and store it
        if self.__hog_features is None:
            sp = self.__search_params
            self.__hog_features = self.__camera_image.hog_features_list(
                self.image, sp.hog_channel, sp.orient,
                sp.pix_per_cell, sp.cell_per_block, sp.color_space)

        return self.__hog_features

    @jit(cache=True)
    def hog_image(self, channel):
        image = self.image
        orient = self.__search_params.orient
        pix_per_cell = self.__search_params.pix_per_cell
        cell_per_block = self.__search_params.cell_per_block

        _, hog_image = hog(image[:, :, channel], orientations=orient,
                           pixels_per_cell=(pix_per_cell, pix_per_cell),
                           cells_per_block=(
            cell_per_block, cell_per_block), transform_sqrt=False,
            visualise=True, feature_vector=False)
        return hog_image

    def window_hog_features(self, window):
        # add all the features from all hog channel windows
        features = Features([])
        for hif in self.hog_features_list:
            features += hif.window_hog_features(window)

        return features

    def window_generator(self):
        """xform window points to meet the coordinates of this image slice"""
        wb = self.__window_boxes
        shape = self.__shape
        bb_origin = wb.bounding_box_origin(shape)
        ratio = wb.resize_64x64_ratio(shape)
        for window in wb.windows(shape):
            new_window = np.multiply(np.subtract(
                window, bb_origin), (ratio, ratio)).astype(int)
            yield WindowBoxSlice(window, new_window)
