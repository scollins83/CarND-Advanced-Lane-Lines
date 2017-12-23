import numpy as np
import glob
import sys
import logging
import json
import argparse
import os
import cv2

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args(arguments):
    """
    Parses arguments given at the command line.
    :param arguments: Arguments given at the command line
    :return: Dict of variables parsed from the arguments
    """
    parser = argparse.ArgumentParser(description="Trains a behavioral cloning model from a given training file set.")
    parser.add_argument('-c', '--configuration', help="File path configuration file", required=True,
                        dest='config')

    return vars(parser.parse_args(arguments))

def load_config(config_name):
    """
    loads a json config file and returns a config dictionary
    """
    with open(config_name) as config_file:
        configuration = json.load(config_file)
        return configuration

def create_object_points(height, width, depth, data_type=np.float32):
    """

    :param height:
    :param width:
    :param depth:
    :param data_type:
    :return:
    """
    return np.zeros((height*width, depth), data_type)


def modify_object_points(object_points, height, width, slicer,
                         reshape_0, reshape_1):
    """
    Provides modification of initial object points.
    :param object_points:
    :param height:
    :param width:
    :param slicer:
    :param reshape_0:
    :param reshape_1:
    :return:
    """
    object_points[:, :slicer] = np.mgrid[0:width, 0:height].T.reshape(reshape_0, reshape_1)
    return object_points


def get_calibration_file_paths(path):
    """

    :param path:
    :return:
    """
    # TODO: Modify this so it can accept a directory.
    return glob.glob(path)

def read_image(path):
    """
    Read an individual image. Wrapper for OpenCV 'imread'
    :param path: Image Path
    :return: Array of images.
    """
    return cv2.imread(path)

def convert_color_image_to_grayscale(color_image_array):
    """
    Converts a three-channel color image array to single-channel grayscale.
    Wrapper for OpenCV cvtColor.
    :param image: 3-channel color image array
    :return: 1-channel grayscale image array of the input image
    """
    return cv2.cvtColor(color_image_array, cv2.COLOR_BGR2GRAY)

def find_chessboard_corners(image, pattern_width, pattern_height, pattern_corners):
    """
    Function to find chessboard corners for calibration.
    Wrapper of OpenCV findChessboardCorners.
    :param image: Grayscale image array.
    :param pattern_width: Number of corners wide for calibration.
    :param pattern_height: Number of corners high for calibration.
    :param pattern_corners: Number of corners to prepass for calibration.
    :return: Indicate a return value, and also the array of corners found, in pixels.
    """
    if pattern_corners == "None":
        input_corners = None
    else:
        input_corners = pattern_corners

    return cv2.findChessboardCorners(image, (pattern_width,
                                             pattern_height), input_corners)



if __name__ == "__main__":

    # Set TensorFlow logging so it isn't so verbose.
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    # Load the necessary parameters
    args = parse_args(sys.argv[1:])
    config = load_config(args['config'])

    calibration_file_pattern = config['calibration_file_pattern']

    sys.exit(0)