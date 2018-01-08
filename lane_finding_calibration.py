import numpy as np
import glob
import sys
import logging
import json
import argparse
import os
import cv2
import pickle

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


def get_image_file_paths(path):
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


def save_calibration_images_with_points(original_image, original_image_file_name, width, height, corners, ret,
                                        calibration_pattern_image_directory):
    """

    :param original_image:
    :param height:
    :param width:
    :param corners:
    :param ret:
    :return:
    """
    img = cv2.drawChessboardCorners(original_image, (width, height), corners, ret)
    cv2.imshow('img', img)
    image_name = original_image_file_name.split('/')[-1]
    write_filename = image_name.replace('.', '_cal.')
    write_filename = calibration_pattern_image_directory + '/' + write_filename
    cv2.imwrite(write_filename, img)
    cv2.waitKey(500)


def get_calibration_object_and_image_points(image_list_pattern, height, width, depth, slicer, reshape_0, reshape_1,
                                            pattern_corners, calibration_pattern_image_directory):
    """

    :param image_list_pattern: Glob pattern for finding calibration images.
    :param height: Height in number of corner points.
    :param width: Width in number of corner points.
    :param depth: Number of original color channels.
    :param slicer: Value of the slicer for setting up initial object point array.
    :param reshape_0: First reshape value for setting up initial object point array.
    :param reshape_1: Second reshape value for setting up initial object point array.
    :param pattern_corners: Corners to pre-feed finding the checkerboard corners.
    :return:
    """
    object_points_list = []
    image_points_list = []

    object_points = create_object_points(height, width, depth)
    object_points = modify_object_points(object_points, height, width, slicer, reshape_0, reshape_1)

    image_paths = get_image_file_paths(image_list_pattern)

    for file_name in image_paths:
        image = read_image(file_name)
        gray_image = convert_color_image_to_grayscale(image)

        ret, corners = find_chessboard_corners(gray_image, width, height, pattern_corners)

        if ret == True:
            object_points_list.append(object_points)
            image_points_list.append(corners)

            # Draw and display corners
            save_calibration_images_with_points(image, file_name, width, height, corners, ret,
                                                calibration_pattern_image_directory)

    cv2.destroyAllWindows()

    return object_points_list, image_points_list

def calibrate_camera(object_points, image_points, image_size):
    """

    :param object_points:
    :param image_points:
    :param image_size:
    :return:
    """
    _ , camera_matrix, distortion_coeffs, _, _ = cv2.calibrateCamera(object_points,
                                                                     image_points,
                                                                     image_size,
                                                                     None,
                                                                     None)
    calibration_dict = {}
    calibration_dict['camera_matrix'] = camera_matrix
    calibration_dict['distortion_coeffs'] = distortion_coeffs

    return calibration_dict


def save_camera_calibration(calibration_dict, save_path):
    """

    :param calibration_dict:
    :param save_path:
    :return:
    """
    with open(save_path, 'wb') as file:
        pickle.dump(calibration_dict, file)


if __name__ == "__main__":

    # Load the necessary parameters
    args = parse_args(sys.argv[1:])
    config = load_config(args['config'])

    # Get the initial object points
    object_points_list, \
    image_points_list = get_calibration_object_and_image_points(config['calibration_file_pattern'],
                                                                config['calibration_pattern_size_height'],
                                                                config['calibration_pattern_size_width'],
                                                                config['calibration_object_channels'],
                                                                config['calibration_object_slice'],
                                                                config['calibration_object_reshape_0'],
                                                                config['calibration_object_reshape_1'],
                                                                config['calibration_pattern_size_corners'],
                                                                config['calibration_pattern_image_directory'])

    # Take an image and calibrate the camera
    image = read_image(config['calibration_image'])
    image_size = (image.shape[1], image.shape[0])
    calibration_dict = calibrate_camera(object_points_list, image_points_list, image_size)
    save_camera_calibration(calibration_dict, config['calibration_save_path'])

    sys.exit(0)