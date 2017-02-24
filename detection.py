import typing
import asyncio
import cv2
import numpy as np

from scipy.ndimage.measurements import label
from scipy.ndimage import generate_binary_structure
from scipy import ndimage

from features import Features, BinSpatialFeatures
from features import ColorHistFeatures, HogImageFeatures
from boxes import WindowBoxes, WindowBoxSlice, draw_boxes
from images import CameraImage, ImageSlice
from search import SearchParams
from collections import deque


class VehicleDetection(object):
    """Vehicle Detection class
    holds all things to do with vehicle detection """

    def __init__(self, search_params, height, width, loop=None,
                 heatmap_history_max=5):
        self.__height = height  # of images being used
        self.__width = width  # of image beinf used
        self.__search_params = search_params  # for all objects
        self.__window_boxes = WindowBoxes(height, width)
        self.__heatmap_history_max = heatmap_history_max

        self.__image = None
        self.__camera_image = None
        self.__hot_windows = None
        self.__heatmap = None
        self.__heatmap_history = deque([])
        self.__labels = None
        self.__image_count = 0

        # if we dont have a coroutine loop passed in create one
        if loop is None:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        self.__loop = loop

    @property
    def img_height(self):
        return self.__height

    @property
    def img_width(self):
        return self.__width

    @property
    def loop(self):
        return loop

    @property
    def image(self):
        return self.__image

    @property
    def window_boxes(self):
        return self.__window_boxes

    @property
    def camera_image(self):
        return self.__camera_image

    @property
    def search_params(self):
        return self.__search_params

    @image.setter
    def image(self, image):
        # make sure its a bgr image being passed in
        self.__image = image
        self.__image_count += 1
        self.__camera_image = CameraImage(image)

        # new image so reset these so they are recreated next access
        self.__hot_windows = None
        self.__heatmap = None
        self.__labels = None

    def _queue_to_history(self, heatmap):
        self.__heatmap_history.append(heatmap)
        # only keep so many heatmaps, removed old ones [n-3,n-2,n-1,n]
        if self.heatmap_history_count > self.__heatmap_history_max:
            self.__heatmap_history.popleft()

    @property
    def heatmap_history_count(self):
        return len(self.__heatmap_history)

    @property
    def hot_windows(self):
        if self.__hot_windows is None:
            self.__hot_windows = self.hot_windows_search()
        return self.__hot_windows

    @property
    def heatmap(self):
        if self.__heatmap is None:
            self.__heatmap = self.build_heatmap()
        return self.__heatmap

    @property
    def heatmap_history(self):
        # if the heatmap hasnt yet been rebuilt since the last image
        # we need to do so first. It will appeend it to the history
        if self.__heatmap is None:
            self.__heatmap = self.build_heatmap()

        return np.array(self.__heatmap_history)

    @property
    def labels(self):
        if self.__labels is None:
            hh = np.array(self.heatmap_history)
            # create a structure for connectivity
            s = generate_binary_structure(hh.ndim, hh.ndim)
            self.__labels = label(hh, s)
        return self.__labels

    @property
    def labelled_boxes(self):
        return self._extract_labeled_bboxes()

    @property
    def box_variance(self):
        lbl, nlbl = self.labels
        return ndimage.variance(self.heatmap_history, lbl,
                                index=np.arange(1, nlbl+1))

    @property
    def heatmap_decorated(self):

        heatmap_ff = np.array(self.heatmap).astype(np.uint8)
        heatmap_ff[heatmap_ff > 0] = 255
        zeros = np.zeros_like(heatmap_ff)
        heatmap_color = np.dstack((heatmap_ff, zeros, zeros))

        result = cv2.addWeighted(self.boxes_decorated,
                                 1, heatmap_color, 0.3, 0)
        return result

    @property
    def boxes_decorated(self):
        # TODO change this back to bgr image - ok whilst testing
        window_img = draw_boxes(self.camera_image.rgb,
                                self.hot_windows, color=(0, 0, 255), thick=2)
        return window_img

    @property
    def result(self):
        window_img = draw_boxes(self.__image, self.labelled_boxes,
                                color=(255, 0, 0), thick=2)
        return window_img

    @staticmethod
    def single_window_features(image_slice: ImageSlice,
                               wbs: WindowBoxSlice,
                               search_params: SearchParams):
        features = Features([])
        window = wbs.bbox_slice  # use the slice window bounding box

        # submit in parallel
        if search_params.hog_feat is True:
            features += image_slice.window_hog_features(window)

        if search_params.spatial_feat is True:
            features += image_slice.window_bin_spatial_features(window)

        if search_params.hist_feat is True:
            features += image_slice.window_color_hist_features(window)

        float_values = features.values.astype(np.float64)
#         print("window: {} {} float_values.shape {}".format (
# window, image_slice.shape,  float_values.shape))

        return float_values

    @staticmethod  # not used but left in for testing
    def search_windows_generator(camera_image: CameraImage,
                                 window_boxes: WindowBoxes,
                                 search_params: SearchParams):
        for shape in window_boxes.shape_keys:
            # the image slice processes the image for the window shape
            image_slice = ImageSlice(
                camera_image, window_boxes, search_params, shape)

            for wbs in image_slice.window_generator():
                # print("yielding: ", shape, wbs.bbox, wbs.bbox_slice)
                yield (image_slice, wbs, shape)

    def hot_windows_search(self):

        camera_image = self.camera_image
        window_boxes = self.window_boxes
        search_params = self.search_params
        hot_windows = []

        async def gather_features_for_prediction(image_slice: ImageSlice,
                                                 wbs: WindowBoxSlice,
                                                 search_params: SearchParams):
            features = Features([])
            gather_co_list = []
            window = wbs.bbox_slice

            async def gather_hog_features(window):
                return image_slice.window_hog_features(window)

            async def gather_spatial_features(window):
                return image_slice.window_bin_spatial_features(window)

            async def gather_color_hist_features(window):
                return image_slice.window_color_hist_features(window)

            if search_params.hog_feat is True:
                gather_co_list.append(gather_hog_features(window))

            if search_params.spatial_feat is True:
                gather_co_list.append(gather_spatial_features(window))

            if search_params.hist_feat is True:
                gather_co_list.append(gather_color_hist_features(window))

            for feature in await asyncio.gather(*gather_co_list):
                features += feature

            float_values = features.values.astype(np.float64)
            return float_values

        async def window_prediction(image_slice: ImageSlice,
                                    wbs: WindowBoxSlice):
            clf = self.search_params.clf
            X_scaler = self.search_params.X_scaler

            # only about 20 milliseconds difference betwween the two approaches
            # features = VehicleDetection.single_window_features(image_slice,
            #                                                    wbs,
            #                                                    search_params)
            features = await gather_features_for_prediction(image_slice,
                                                            wbs,
                                                            search_params)
            # print(features.shape, wbs.bbox, wbs.bbox_slice)
            try:
                scaled_features = X_scaler.transform(features.reshape(1, -1))

                prediction = clf.predict(scaled_features)
                if prediction == 1:
                    # print(wbs.bbox)
                    hot_windows.append(wbs.bbox)
            except ValueError as exc:
                fs = "{} features.shape {} wbs.bbox {} wbs.bbox_slice"
                print(fs.format(exc, wbs.bbox, wbs.bbox_slice))

        async def predict_hot_boxes(shape):
            image_slice = ImageSlice(
                camera_image, window_boxes, search_params, shape)
            await asyncio.gather(*[window_prediction(image_slice, wbs)
                                   for wbs in image_slice.window_generator()])

        async def window_box_predictions(camera_image: CameraImage,
                                         window_boxes: WindowBoxes):
            await asyncio.gather(*[predict_hot_boxes(shape)
                                   for shape in window_boxes.shape_keys])

        loop = asyncio.get_event_loop()
        loop.run_until_complete(window_box_predictions(
            camera_image, window_boxes))

        # print("hot_windows:", hot_windows)

        return hot_windows

    def build_heatmap(self):

        heatmap = np.zeros_like(self.image[:, :, 0]).astype(np.float)

        def add_heat(heatmap, bbox_list):
            for box in bbox_list:
                # Add += 1 for all pixels inside each bbox
                # Assuming each "box" takes the form ((x1, y1), (x2, y2))
                heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
            return heatmap

        def apply_threshold(heatmap, threshold):
            # Zero out pixels below the threshold
            heatmap[heatmap <= threshold] = 0
            return heatmap

        heatmap = add_heat(heatmap, self.hot_windows)

        heatmap = np.clip(heatmap, 0, 255)

        heatmap = apply_threshold(heatmap, 4)

        # standardise heatmap -
        heatmap_std = heatmap.std(ddof=1)
        if heatmap_std != 0.0:
            heatmap = (heatmap-heatmap.mean())/heatmap_std

        heatmap = apply_threshold(heatmap, np.max([heatmap.std(), 1]))

        # add this heatmap to the queue
        self._queue_to_history(heatmap)
        return heatmap

    @staticmethod
    def draw_boxes(img, bboxes, color=(255, 0, 0), thick=6):
        # Make a copy of the image
        imcopy = np.copy(img)
        # Iterate through the bounding boxes
        for bbox in bboxes:
            # Draw a rectangle given bbox coordinates
            cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
        # Return the image copy with boxes drawn
        return imcopy

    @property
    def label_box_planes(self):
        labels = self.labels
        planes = []

        for car_number in range(1, labels[1] + 1):
            nonzero = (labels[0] == car_number).nonzero()
            nonzeroz = np.array(nonzero[0])
            planes.append((np.min(nonzeroz), np.max(nonzeroz)))

        return planes

    def _extract_labeled_bboxes(self):
        labels = self.labels
        box_variance = self.box_variance

        bboxes = []
        for car_number in range(1, labels[1] + 1):
            # if just a few point found in the heatmap ignore
            if labels[1] == 1 and box_variance[car_number-1] < 0.1:
                continue
            elif box_variance[car_number-1] < 1.5:
                continue

            # Find pixels with each car_number label value
            nonzero = (labels[0] == car_number).nonzero()
            # Identify x and y values of those pixels
            nonzeroz = np.array(nonzero[0])
            nonzeroy = np.array(nonzero[1])
            nonzerox = np.array(nonzero[2])

            nonzerox_min = np.min(nonzerox)
            nonzerox_max = np.max(nonzerox)
            nonzeroy_min = np.min(nonzeroy)
            nonzeroy_max = np.max(nonzeroy)
            nonzeroz_min = np.min(nonzeroz)
            nonzeroz_max = np.max(nonzeroz)

            # only add if they appear in contiguous planes
            nplane_min_threshold = self.__heatmap_history_max - 2
            # planes connected via label function and ndims of heatmap
            # they start at 0 so add 1
            nplanes = nonzeroz_max-nonzeroz_min+1

            if nplanes >= nplane_min_threshold:
                bbox = ((nonzerox_min, nonzeroy_min),
                        (nonzerox_max, nonzeroy_max))
                bboxes.append(bbox)
        return bboxes
