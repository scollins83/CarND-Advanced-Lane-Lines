import numpy as np

def create_object_points(height, width, depth, type=np.float32):
    """

    :param height:
    :param width:
    :param depth:
    :param type:
    :return:
    """
    return np.zeros((height*width, depth), type)