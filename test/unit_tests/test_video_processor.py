import unittest
import lane_finding_calibration as lfc
import video_processor as ip
import cv2
import pickle
import numpy as np


class TestImageProcessing(unittest.TestCase):

    def setUp(self):
        self.config = lfc.load_config('test_img_processing_configuration.json')
        self.test_image = cv2.imread('../../test_images/ckbd_test_image.jpg')
        with open(self.config['object_points_pickle'], 'rb') as o_file:
            self.object_points = pickle.load(o_file)
        with open(self.config['image_points_pickle'], 'rb') as i_file:
            self.image_points = pickle.load(i_file)
        self.calibration_dict = {'camera_matrix': np.array([[1.29383719e+03,
                                                             0.00000000e+00,
                                                             3.24452248e+02],
                                                            [0.00000000e+00,
                                                             1.28959394e+03,
                                                             2.70105841e+02],
                                                            [0.00000000e+00,
                                                             0.00000000e+00,
                                                             1.00000000e+00]]),
                                 'distortion_coeffs': np.array([[-0.5534137,
                                                                 0.56037615,
                                                                 0.00537364,
                                                                 0.01884776,
                                                                 -0.3336068]])}


    # INCLUDE THRESHOLDING FUNCTIONS AND WINDOW_MASK
    def test_hls_select(self):
        pass

    def test_abs_sobel_thresh(self):
        pass

    def test_window_mask(self):
        pass

    def tearDown(self):
        del self.object_points
        del self.image_points
        del self.test_image
        del self.calibration_dict
        del self.config


if __name__ == '__main__':
    unittest.main()
