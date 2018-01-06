import unittest
import lane_finding_calibration as lfc
import image_preprocessing as ip
import os
import glob
import cv2


class TestImagePreprocessing(unittest.TestCase):

    def setUp(self):
        self.config = lfc.load_config('test_configuration.json')
        self.test_image = cv2.imread('../../test_images/test_image.png')
        self.object_points, self.image_points = \
            lfc.get_calibration_object_and_image_points(self.config['calibration_file_pattern'],
                                                        self.config['calibration_pattern_size_height'],
                                                        self.config['calibration_pattern_size_width'],
                                                        self.config['calibration_object_channels'],
                                                        self.config['calibration_object_slice'],
                                                        self.config['calibration_object_reshape_0'],
                                                        self.config['calibration_object_reshape_1'],
                                                        self.config['calibration_pattern_size_corners'],
                                                        self.config['calibration_pattern_image_directory'])


    def test_undistort_image(self):
        undistorted_image = ip.undistort_image(self.test_image, self.object_points, self.image_points)
        cv2.imshow('Undistorted Test Image', undistorted_image)
        self.assertEqual(undistorted_image.shape, self.test_image.shape)

    def tearDown(self):
        del self.object_points
        del self.image_points
        del self.test_image

        saved_img_list = glob.glob(self.config['calibration_pattern_image_directory'] + '/*.jpg')
        if len(saved_img_list) > 0:
            [os.remove(path) for path in saved_img_list]

        del self.config


if __name__ == '__main__':
    unittest.main()
