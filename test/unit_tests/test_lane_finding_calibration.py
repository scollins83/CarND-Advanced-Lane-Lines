import unittest
import numpy as np
import lane_finding_calibration as lf
import glob
import os
import pickle
import cv2


class TestLaneFindingCalibration(unittest.TestCase):

    def setUp(self):
        self.config = lf.load_config('test_calibration_configuration.json')

    def test_create_object_points(self):
        grid_height = 6
        grid_width = 9
        grid_depth = 3
        tester_object = np.zeros((grid_height*grid_width, grid_depth), np.float32)
        grid_object = lf.create_object_points(grid_height, grid_width, grid_depth)
        self.assertEqual(tester_object.shape, grid_object.shape)

    def test_modify_object_points(self):
        grid_height = 6
        grid_width = 9
        grid_depth = 3
        slice_of_object = 2
        reshape_0 = -1
        reshape_1 = 2

        tester_object = np.zeros((grid_height*grid_width, grid_depth), np.float32)
        modified_object = lf.modify_object_points(tester_object, grid_height,
                                                  grid_width, slice_of_object,
                                                  reshape_0, reshape_1)
        self.assertListEqual(list(modified_object[-1]), [8., 5., 0.])

    def test_get_calibration_file_paths(self):
        path = self.config['calibration_file_pattern']
        image_list = lf.get_calibration_file_paths(path)
        self.assertEqual(len(image_list), 20)

    def test_read_image(self):
        path = self.config['calibration_file_pattern']
        image_list = lf.get_calibration_file_paths(path)
        image = lf.read_image(image_list[0])
        self.assertEqual(len(image.shape), 3)
        self.assertEqual(image.shape[0], 720)
        self.assertEqual(image.shape[1], 1280)
        self.assertEqual(image.shape[2], 3)

    def test_convert_color_image_to_grayscale(self):
        path = self.config['calibration_file_pattern']
        image_list = lf.get_calibration_file_paths(path)
        image = lf.read_image(image_list[0])
        gray_image = lf.convert_color_image_to_grayscale(image)
        self.assertEqual(len(gray_image.shape), 2)
        self.assertEqual(gray_image.shape[0], 720)
        self.assertEqual(gray_image.shape[1], 1280)

    def test_find_chessboard_corners(self):
        path = self.config['calibration_file_pattern']
        image_list = lf.get_calibration_file_paths(path)
        image = lf.read_image(image_list[0])
        gray_image = lf.convert_color_image_to_grayscale(image)
        ret, corners = lf.find_chessboard_corners(gray_image,
                                                  self.config['calibration_pattern_size_width'],
                                                  self.config['calibration_pattern_size_height'],
                                                  self.config['calibration_pattern_size_corners'])
        self.assertFalse(ret)
        self.assertEqual(len(corners.shape), 3)
        self.assertEqual(corners.shape[0], 53)
        self.assertEqual(corners.shape[1], 1)
        self.assertEqual(corners.shape[2], 2)

    def test_save_calibration_images_with_points(self):
        path = self.config['calibration_file_pattern']
        image_list = lf.get_calibration_file_paths(path)
        image = lf.read_image(image_list[1])
        gray_image = lf.convert_color_image_to_grayscale(image)
        ret, corners = lf.find_chessboard_corners(gray_image,
                                                  self.config['calibration_pattern_size_width'],
                                                  self.config['calibration_pattern_size_height'],
                                                  self.config['calibration_pattern_size_corners'])
        original_saved_img_list = glob.glob(self.config['calibration_pattern_image_directory'] + '/*.jpg')
        self.assertEqual(len(original_saved_img_list), 0)
        lf.save_calibration_images_with_points(image, image_list[0], self.config['calibration_pattern_size_width'],
                                               self.config['calibration_pattern_size_height'], corners, ret,
                                               self.config['calibration_pattern_image_directory'])
        after_saved_img_list = glob.glob(self.config['calibration_pattern_image_directory'] + '/*.jpg')
        self.assertEqual(len(after_saved_img_list), 1)
        os.remove(after_saved_img_list[0])

    def test_get_calibration_object_and_image_points(self):
        calibration_object_points, calibration_image_points = \
            lf.get_calibration_object_and_image_points(self.config['calibration_file_pattern'],
                                                       self.config['calibration_pattern_size_height'],
                                                       self.config['calibration_pattern_size_width'],
                                                       self.config['calibration_object_channels'],
                                                       self.config['calibration_object_slice'],
                                                       self.config['calibration_object_reshape_0'],
                                                       self.config['calibration_object_reshape_1'],
                                                       self.config['calibration_pattern_size_corners'],
                                                       self.config['calibration_pattern_image_directory'])

        self.assertIsNotNone(calibration_object_points)
        self.assertIsNotNone(calibration_image_points)
        saved_img_list = glob.glob(self.config['calibration_pattern_image_directory'] + '/*.jpg')
        for path in saved_img_list:
            os.remove(path)

    def test_calibrate_camera(self):
        image = cv2.imread('../../test_images/test_image.png')
        image_size = (image.shape[1], image.shape[0])
        with open('test_object_points.pkl', 'rb') as f:
            object_points = pickle.load(f)
        with open('test_image_points.pkl', 'rb') as g:
            image_points = pickle.load(g)
        camera_calibration_dict = lf.calibrate_camera(object_points,
                                                      image_points,
                                                      image_size)
        test_dict = {'camera_matrix': np.array([[1.29383719e+03,
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
        self.assertEqual(camera_calibration_dict['camera_matrix'].all(),
                         test_dict['camera_matrix'].all())
        self.assertEqual(camera_calibration_dict['distortion_coeffs'].all(),
                         test_dict['distortion_coeffs'].all())

    def test_save_camera_calibration(self):
        test_dict = {'camera_matrix': np.array([[1.29383719e+03,
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
        lf.save_camera_calibration(test_dict, self.config['calibration_save_path'])
        saved_path = glob.glob(self.config['calibration_save_path'])
        self.assertEqual(len(saved_path), 1)
        for path in saved_path:
            os.remove(path)

    def tearDown(self):
        del self.config


if __name__ == '__main__':
    unittest.main()
