import numpy as np
import cv2
import typing


# Here is your draw_boxes function from the previous exercise
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imgcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imgcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imgcopy


class WindowBoxSlice(typing.NamedTuple("GenWindowSlice",
                                       [('bbox', tuple),
                                        ('bbox_slice', tuple)]
                                       )):
    pass


class WindowBoxes(object):
    """WindowBoxes class - all things to do with boxes"""

    def __init__(self, height, width):

        self.__sz_dict = self._spatial_sizes_dict()
        self.__wb_dict = self.build_window_boxes(height, width)
        self.__height = height
        self.__width = width
        self.__bounding_box_dict = {key: None for key, _
                                    in self.__wb_dict.items()}

    @property
    def spatial_sizes_dict(self):
        return self.__sz_dict

    @property
    def shape_keys(self):
        return [key for key, _ in self.__wb_dict.items()]

    @property
    def window_boxes_dict(self):
        return self.__wb_dict

    @property
    def box_colour_dict(self):
        box_color = {}
        box_color["small"] = (0, 0, 128)
        box_color["smallish"] = (0, 0, 256)
        box_color["medium"] = (0, 256, 0)
        box_color["large"] = (128, 0, 0)
        box_color["max"] = (256, 0, 0)
        return box_color

    @property
    def height(self):
        return self.__height

    @property
    def width(self):
        return self.__width

    def windows(self, shape):
        return self.window_boxes_dict[shape]

    def bounding_box(self, shape):
        bb = self.__bounding_box_dict[shape]
        if bb is None:
            pts = np.array(self.window_boxes_dict[shape]).reshape(-1, 2)

            pts_x, pts_y = pts[:, 0], pts[:, 1]

            min_x, max_x = np.min(pts_x), np.max(pts_x)
            min_y, max_y = np.min(pts_y), np.max(pts_y)

            bb = ((min_x, min_y), (max_x, max_y))
            self.__bounding_box_dict[shape] = bb

        return bb

    def bounding_box_origin(self, shape):
        return self.bounding_box(shape)[0]

    @staticmethod
    def _spatial_sizes_dict():
        sz_dict = {}
        sz_dict["small"] = (32, 32)
        sz_dict["smallish"] = (64, 64)
        sz_dict["medium"] = (128, 128)
        sz_dict["large"] = (256, 256)
        sz_dict["max"] = (320, 320)
        return sz_dict

    @staticmethod
    def resize_64x64_ratio(shape):
        size = WindowBoxes._spatial_sizes_dict()[shape][0]
        return 64. / size

    @staticmethod
    def build_window_boxes(height, width, all_shapes=True, shape="small"):
        # Define a function that takes an image,
        # start and stop positions in both x and y,
        # window size (x and y dimensions),
        # and overlap fraction (for both x and y)
        def slide_window(height, width, x_start_stop=[None, None],
                         y_start_stop=[None, None],
                         xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
            # If x and/or y start/stop positions not defined, set to image size
            if x_start_stop[0] is None:
                x_start_stop[0] = 0
            if x_start_stop[1] is None:
                x_start_stop[1] = width
            if y_start_stop[0] is None:
                y_start_stop[0] = 0
            if y_start_stop[1] is None:
                y_start_stop[1] = height
            # Compute the span of the region to be searched
            xspan = x_start_stop[1] - x_start_stop[0]
            yspan = y_start_stop[1] - y_start_stop[0]
            # Compute the number of pixels per step in x/y
            nx_pix_per_step = np.int(xy_window[0] * (1 - xy_overlap[0]))
            ny_pix_per_step = np.int(xy_window[1] * (1 - xy_overlap[1]))
            # Compute the number of windows in x/y
            nx_windows = np.int(xspan / nx_pix_per_step) - 1
            ny_windows = np.int(yspan / ny_pix_per_step) - 1
            # Initialize a list to append window positions to
            window_list = []
            # Loop through finding x and y window positions
            # Note: you could vectorize this step, but in practice
            # you'll be considering windows one by one with your
            # classifier, so looping makes sense
            for ys in range(ny_windows):
                for xs in range(nx_windows):
                    # Calculate window position
                    startx = xs * nx_pix_per_step + x_start_stop[0]
                    endx = startx + xy_window[0]
                    starty = ys * ny_pix_per_step + y_start_stop[0]
                    endy = starty + xy_window[1]
                    # Append window position to list
                    window_list.append(((startx, starty), (endx, endy)))
            # Return the list of windows
            return window_list

        def find_windows(height, width, xc=7, yc=2.5, xy_window=(32, 32),
                         xy_overlap=(0.5, 0.5)):
            y_center = height // 2
            x_center = width // 2
            x, y = xy_window
            y_start = y_center
            if y == 32:
                y_start = y_center + y
            elif y == 64:
                y_start = y_center + np.int(y / 2)
            elif y == 128:
                y_start = y_center + 25
            elif y == 256:
                y_start = y_center
            elif y == 384:
                y_start = y_center
            # y_stop = height - 150  # for bonnet
            # if y <= 128:
            #     y_stop = y_center + np.int(y * yc)
            # windows = slide_window(height, width,
            #                        x_start_stop=[x_center - np.int(x * xc),
            #                                      x_center + np.int(x * xc)],
            #                        y_start_stop=[y_start,
            #                                      y_center + np.int(y * yc)],
            #                        xy_window=xy_window, xy_overlap=xy_overlap)
            # adjusting these windows to fit the project video
            windows = slide_window(height, width,
                                   x_start_stop=[x_center+96,
                                                 x_center + np.int(x * xc)],
                                   y_start_stop=[y_start,
                                                 y_center + np.int(y * yc)],
                                   xy_window=xy_window, xy_overlap=xy_overlap)
            return windows

        sz_dict = WindowBoxes._spatial_sizes_dict()

        window_dict = {}
        window_dict["small"] = find_windows(
            height, width, xc=19.5, yc=6, xy_window=sz_dict['small'],
            xy_overlap=(0.5, 0.5))
        window_dict["smallish"] = find_windows(
            height, width, xc=9.5, yc=2, xy_window=sz_dict['smallish'],
            xy_overlap=(0.7, 0.7))
        window_dict["medium"] = find_windows(
            height, width, xc=4, yc=1.25, xy_window=sz_dict['medium'],
            xy_overlap=(0.9, 0.6))
        # window_dict["large"] = find_windows(
        # height, width, xc=1.5, yc=1, xy_window=sz_dict['large'],
        # xy_overlap=(0.7,0.6))
        # window_dict["max"] = find_windows(
        # height, width, xc=1.9, yc=1, xy_window=sz_dict['max'],
        # xy_overlap=(0.6,0.6))

        # if not all shapes then overwrite with just the shape required
        if all_shapes is not True:
            window_dict = {shape: window_dict[shape]}

        return window_dict

        @staticmethod
        def _spatial_sizes_dict():
            sz_dict = {}
            sz_dict["small"] = (64, 64)
            sz_dict["smallish"] = (96, 96)
            sz_dict["medium"] = (128, 128)
            sz_dict["large"] = (256, 256)
            sz_dict["max"] = (320, 320)
            return sz_dict
