import numpy as np
from PIL import Image
import os, fnmatch
import tensorflow as tf
import datetime
import h5py
import cv2
import time
import SimpleITK as sitk

##############################
## Author: S. Elguindi
##
## Defined variables, flags
##############################

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('inference_size', 256,
                     'Size of input image in model')

flags.DEFINE_integer('num_classes', 8,
                     'Number of classes defined in model')

flags.DEFINE_string('data_dir', '/scratch/inputH5/',
                    'absolute path where patient H5 scans are stored')

flags.DEFINE_string('H5_Name', 'SCAN',
                    'keyword string to find H5 files to run inference on in data_dir')

flags.DEFINE_string('save_dir', '/scratch/outputH5',
                    'absolute path to save output MASKs, typically same folder')

#flags.DEFINE_string('model_path', '/software/model/frozen_inference_graph.pb',
#                    'absolute path to saved model DeepLab Model')

flags.DEFINE_string('model_path', 'L:/Aditi/MR_Prostate_Deeplab_ver2/model/frozen_inference_graph.pb',
                    'absolute path to saved model DeepLab Model')


def normalize_array_8bit(arr):
    norm_arr = np.zeros(np.shape(arr), dtype='uint8')
    norm_arr = cv2.normalize(arr, norm_arr, 0, 255, cv2.NORM_MINMAX)
    return norm_arr


def normalize_array_16bit(arr):
    norm_arr = np.zeros(np.shape(arr), dtype='uint16')
    norm_arr = cv2.normalize(arr, norm_arr, 0, 65535, cv2.NORM_MINMAX)
    return norm_arr


def equalize_array(img, stacked_img, clahe):
    eq_img = clahe.apply(img)
    stacked_img[:, :, 0] = eq_img
    stacked_img[:, :, 1] = eq_img
    stacked_img[:, :, 2] = eq_img

    return eq_img, stacked_img.astype('uint8')


def find(pattern, path):
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name))
    return result


def pad_with(vector, pad_width, iaxis, kwargs):
    pad_value = kwargs.get('padder', 0)
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value
    return vector


def find_file(name, path):
    for root, dirs, files in os.walk(path):
        if name in files:
            return os.path.join(root, name)


def normalize_equalize_smooth_MR(arr, clahe, pct_remove_bot, pct_remove_top):

    length, width, height = np.shape(arr)
    norm_arr = np.zeros(np.shape(arr), dtype='uint16')
    norm_arr = cv2.normalize(arr, norm_arr, 0, 65535, cv2.NORM_MINMAX)
    hist, bin_edges = np.histogram(norm_arr[:], 255)
    cum_sum = np.cumsum(hist.astype('float') / np.sum(hist).astype('float'))
    try:
        clip_value_min = bin_edges[np.min(np.where(cum_sum < pct_remove_bot))]
    except ValueError:
        clip_value_min = 0
    clip_value_max = bin_edges[np.min(np.where(cum_sum > pct_remove_top))]
    norm_arr = np.clip(norm_arr, clip_value_min, clip_value_max)
    norm_eq = np.zeros((length, 3, width, height), dtype='uint16')
    if clahe is not None:
        for ii in range(0, length):
            if ii == 0:
                img_0 = clahe.apply(norm_arr[ii + 0, :, :].astype('uint16'))
                img_1 = clahe.apply(norm_arr[ii + 0, :, :].astype('uint16'))
                img_2 = clahe.apply(norm_arr[ii + 1, :, :].astype('uint16'))
            elif ii == length - 1:
                img_0 = clahe.apply(norm_arr[ii - 1, :, :].astype('uint16'))
                img_1 = clahe.apply(norm_arr[ii, :, :].astype('uint16'))
                img_2 = clahe.apply(norm_arr[ii, :, :].astype('uint16'))
            else:
                img_0 = clahe.apply(norm_arr[ii - 1, :, :].astype('uint16'))
                img_1 = clahe.apply(norm_arr[ii + 0, :, :].astype('uint16'))
                img_2 = clahe.apply(norm_arr[ii + 1, :, :].astype('uint16'))

            norm_eq[ii, 0, :, :] = smooth_image(img_0).astype('uint16')
            norm_eq[ii, 1, :, :] = 255 - smooth_image(img_1).astype('uint16')
            norm_eq[ii, 2, :, :] = smooth_image(img_2).astype('uint16')
    else:
        for ii in range(0, length):
            if ii == 0:
                img_0 = norm_arr[ii + 0, :, :].astype('uint16')
                img_1 = norm_arr[ii + 0, :, :].astype('uint16')
                img_2 = norm_arr[ii + 1, :, :].astype('uint16')
            elif ii == length - 1:
                img_0 = norm_arr[ii - 1, :, :].astype('uint16')
                img_1 = norm_arr[ii, :, :].astype('uint16')
                img_2 = norm_arr[ii, :, :].astype('uint16')
            else:
                img_0 = norm_arr[ii - 1, :, :].astype('uint16')
                img_1 = norm_arr[ii + 0, :, :].astype('uint16')
                img_2 = norm_arr[ii + 1, :, :].astype('uint16')

            norm_eq[ii, 0, :, :] = smooth_image(img_0).astype('uint16')
            norm_eq[ii, 1, :, :] = 255 - smooth_image(img_1).astype('uint16')
            norm_eq[ii, 2, :, :] = smooth_image(img_2).astype('uint16')

    norm_arr_ds = np.zeros(np.shape(norm_eq), dtype='uint8')
    norm_arr_ds = cv2.normalize(norm_eq, norm_arr_ds, 0, 255, cv2.NORM_MINMAX)
    return norm_arr_ds.astype('uint8')


def smooth_image(arr, t_step=0.125, n_iter=3):
    img = sitk.GetImageFromArray(arr)
    img = sitk.CurvatureFlow(image1=img,
                             timeStep=t_step,
                             numberOfIterations=n_iter)
    arr_smoothed = sitk.GetArrayFromImage(img)
    return arr_smoothed


## DeepLabV3 class, uses frozen graph to load weights, make predictions
class DeepLabModel(object):
    """Class to load deeplab model and run inference."""

    INPUT_TENSOR_NAME = 'ImageTensor:0'
    OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
    INPUT_SIZE = FLAGS.inference_size

    def __init__(self, tarball_path):
        """Creates and loads pretrained deeplab model."""
        self.graph = tf.Graph()

        graph_def = None
        # Extract frozen graph from tar archive.
        file_handle = open(tarball_path, 'rb')
        graph_def = tf.GraphDef.FromString(file_handle.read())

        if graph_def is None:
            raise RuntimeError('Cannot find inference graph in tar archive.')

        with self.graph.as_default():
            tf.import_graph_def(graph_def, name='')

        self.sess = tf.Session(graph=self.graph)

    def run(self, image):
        """Runs inference on a single image.

        Args:
            image: A PIL.Image object, raw input image.

        Returns:
            resized_image: RGB image resized from original input image.
            seg_map: Segmentation map of `resized_image`.
        """
        batch_seg_map = self.sess.run(
            self.OUTPUT_TENSOR_NAME,
            feed_dict={self.INPUT_TENSOR_NAME: image})
        seg_map = batch_seg_map
        return image, seg_map


def main(argv):
    #---AI temp testing---
    tf.config.experimental.set_visible_devices([], 'GPU')
    argv.append(r"\\vpensmph\deasylab1\Aditya\mpClinTesting\failed_jobs\MR_Prostate_v2\nii_testing\inputH5")
    argv.append(r"\\vpensmph\deasylab1\Aditya\mpClinTesting\failed_jobs\MR_Prostate_v2\nii_testing\outputH5")
    #---------------------
    num_args = len(argv)
    if num_args == 1: # called from container, so only the filename included in argv by default
        data_path = FLAGS.data_dir
        save_dir = FLAGS.save_dir
        model_path = FLAGS.model_path
    else:
        data_path = argv[1]
        save_dir = argv[2]
        scriptDir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(scriptDir, os.pardir, 'model','frozen_inference_graph.pb')

    infer_size = FLAGS.inference_size
    files = find('*.h5', data_path)
    keyword = FLAGS.H5_Name
    model = DeepLabModel(model_path)

    for filename in files:
        print(filename)
        s = h5py.File(filename, 'r')
        
        # scan = s['scan'][:]
        
        datasetName = list(s.keys())
        if "scan" in datasetName:
            scan = s['/scan'][:]
        else:
            if "OctaveExportScan" in datasetName:
                scan = s['/OctaveExportScan']
                scan = scan['value']
                scan = scan[:]
            else:
                raise Exception("Scan dataset not found.")     
        
        scan_flip = np.flipud(np.rot90(scan, axes=(0, 2)))
        print('Loaded SCAN array...')
        path, file = os.path.split(filename)
        print(file.replace(keyword, 'MASK'))
        height, width, length = np.shape(scan_flip)

        print('Computing DeepLab Model...')
        start_time = time.time()

        scan_norm = normalize_equalize_smooth_MR(scan, None, 0.05, 0.95)
        mask = np.zeros((height, width, scan_norm.shape[0]), dtype=np.uint8)
        stacked_img_1 = np.zeros((1, height, width, 3), dtype=np.uint8)
        for i in range(0, length):
            stacked_img_1[0, :, :, :] = scan_norm[i, :, :, :].transpose(2, 1, 0)
            r_im, seg = model.run(stacked_img_1)
            mask[:, :, i] = seg[0, :, :]

        print("--- inference took %s seconds ---" % (time.time() - start_time))
        maskfilename = file.replace(keyword, 'MASK')

        with h5py.File(os.path.join(save_dir, maskfilename), 'w') as hf:
            hf.create_dataset("mask", data=mask[:, :, 0:length])


if __name__ == '__main__':
    tf.app.run()
