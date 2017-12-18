import numpy as np
import glob


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
    object_points[:,:slicer] = np.mgrid[0:width, 0:height].T.reshape(reshape_0, reshape_1)
    return object_points


def get_calibration_file_paths(path):
    """

    :param path:
    :return:
    """
    # TODO: Modify this so it can accept a directory.
    return glob.glob(path)
