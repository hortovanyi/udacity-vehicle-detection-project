from io import BytesIO
import cv2
import numpy as np
import glob
from tqdm import tqdm
from IPython.display import Image
import matplotlib as mpl
from concurrent.futures import ThreadPoolExecutor, as_completed
from moviepy.editor import VideoFileClip
from features import Features
from features import ColorHistFeatures, HogImageFeatures


def load_test_images(glob_regex='test_images/*.jpg'):
    images = []
    files = []

    for f in glob.glob(glob_regex):
        img = cv2.imread(f)
        # img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append(img)
        files.append(f)
        # print(f,img.shape)

    return images, files


def load_test_video(file_name='test_video.mp4'):
    vimages = []
    vframes = []
    count = 0
    clip = VideoFileClip(file_name)
    for img in clip.iter_frames(progress_bar=True):
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        vimages.append(img)
        vframes.append("%s - %d" % (file_name, count))
        count += 1

    return vimages, vframes


def arr2img(arr):
    """Display a 2- or 3-d numpy array as an image."""
    if arr.ndim == 2:
        format, cmap = 'png', mpl.cm.gray
    elif arr.ndim == 3:
        format, cmap = 'jpg', None
    else:
        raise ValueError("Only 2- or 3-d arrays can be displayed as images.")
    # Don't let matplotlib autoscale the color range so we can control
    # overall luminosity
    vmax = 255 if arr.dtype == 'uint8' else 1.0
#     vmax=1.0
    with BytesIO() as buffer:
        mpl.image.imsave(buffer, arr, format=format, cmap=cmap,
                         vmin=0, vmax=vmax)
        out = buffer.getvalue()
    return Image(out)


def load_images(glob_regex='var/non-vehicles/Extras/*.png'):

    images = []
    files = []
    for f in glob.glob(glob_regex):
        img = cv2.imread(f)
        images.append(img)
        files.append(f)

    return np.array(images), np.array(files)


def load_car_not_car_images(root_path='var',
                            non_vehicle_sub_dirs=['Extras', 'GTI'],
                            vehicle_sub_dirs=['GTI_Far', 'GTI_Left',
                                              'GTI_MiddleClose', 'GTI_Right',
                                              'KITTI_extracted']):
    non_vehicle_paths = [root_path + '/non-vehicles/' + p
                         for p in non_vehicle_sub_dirs]
    vehicle_paths = [root_path + '/vehicles/' + p
                     for p in vehicle_sub_dirs]

    def images_from_path(path):
        images, files = load_images(glob_regex=path + '/*.png')
        print(path, images.shape, len(files))
        return images

    non_vehicle_images = np.concatenate([images_from_path(p)
                                         for p in non_vehicle_paths], axis=0)
    vehicle_images = np.concatenate([images_from_path(p)
                                     for p in vehicle_paths], axis=0)

    return vehicle_images, non_vehicle_images


def extract_color_features(imgs, cspace='BGR', spatial_size=(32, 32),
                           hist_bins=32, hist_range=(0, 256)):
    # Create a list to append feature vectors to
    all_features = []

    def feature_extract(image, cspace='BGR', spatial_size=(32, 32),
                        hist_bins=32, hist_range=(0, 256)):

        # apply color conversion if other than 'RGB'
        if cspace != 'BGR':
            if cspace == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            elif cspace == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2LUV)
            elif cspace == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
            elif cspace == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
            elif cspace == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        else:
            feature_image = np.copy(image)

        # bsf = BinSpatialFeatures(feature_image, size=spatial_size)
        chf = ColorHistFeatures(
            feature_image, nbins=hist_bins, bins_range=hist_range)
        return chf.values

    pbar = tqdm(total=len(imgs))
    with ThreadPoolExecutor() as executor:
        extract_futures = {executor.submit(feature_extract, img,
                                           cspace, spatial_size,
                                           hist_bins, hist_range):
                           img for img in imgs}

        for future in as_completed(extract_futures):
            img = extract_futures[future]
            try:
                features = future.result()
                pbar.update(1)
            except Exception as exc:
                print('image feature extract generated an exception: %s'
                      % (exc))
                import sys
                import traceback
                exc_type, exc_value, exc_traceback = sys.exc_info()
                print(repr(traceback.format_tb(exc_traceback)))
                raise exc
            else:
                all_features.append(features)

    pbar.close()
    return all_features


def extract_hog_features(imgs, cspace='BGR', orient=9,
                         pix_per_cell=8, cell_per_block=2, hog_channel=0,
                         spatial_size=(32, 32),
                         hist_bins=32, hist_range=(0, 256)):
    # Create a list to append feature vectors to
    all_features = []

    def feature_extract(image, cspace='BGR', orient=9,
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_size=(32, 32),
                        hist_bins=32, hist_range=(0, 256)):

        # apply color conversion if other than 'RGB'
        if cspace != 'BGR':
            if cspace == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            elif cspace == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2LUV)
            elif cspace == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
            elif cspace == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
            elif cspace == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        else:
            feature_image = np.copy(image)

        # bsf = BinSpatialFeatures(feature_image, size=spatial_size)
        # chf = ColorHistFeatures(
        #     feature_image, nbins=hist_bins, bins_range=hist_range)

        # Call get_hog_features()
        hog_features = Features([])
        if hog_channel == 'ALL':
            for channel in range(feature_image.shape[2]):
                hog_features += HogImageFeatures(
                    feature_image[:, :, channel],
                    orient, pix_per_cell, cell_per_block)
        else:
            hog_features += HogImageFeatures(
                feature_image[:, :, hog_channel],
                orient, pix_per_cell, cell_per_block)

#         return np.concatenate((hog_features.values, chf.values))
        # features = hog_features + chf
        return hog_features.features

    pbar = tqdm(total=len(imgs))
    with ThreadPoolExecutor() as executor:
        extract_futures = {executor.submit(feature_extract, img, cspace,
                                           orient,
                                           pix_per_cell, cell_per_block,
                                           hog_channel, spatial_size,
                                           hist_bins, hist_range):
                           img for img in imgs}

        for future in as_completed(extract_futures):
            img = extract_futures[future]
            try:
                features = future.result()
                pbar.update(1)
            except Exception as exc:
                print('feature extract generated an exception: %s' % (exc))
                import sys
                import traceback
                exc_type, exc_value, exc_traceback = sys.exc_info()
                print(repr(traceback.format_tb(exc_traceback)))
                raise exc
            else:
                all_features.append(features.values)
    pbar.close()
    return all_features
