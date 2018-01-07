import unittest
import lane_finding_calibration as lfc
import image_processing as ip
import cv2
import pickle


class TestImageProcessing(unittest.TestCase):

    def setUp(self):
        self.config = lfc.load_config('test_img_processing_configuration.json')
        self.test_image = cv2.imread('../../test_images/test_image.png')
        with open(self.config['calibration_pickle'], 'rb') as file:
            self.calibration_dict = pickle.load(file)
        with open(self.config['object_points_pickle'], 'rb') as file:
            self.object_points = pickle.load(file)
        with open(self.config['image_points_pickle'], 'rb') as file:
            self.image_points = pickle.load(file)

    def test_undistort_image(self):
        undistorted_image = ip.undistort_image(self.test_image, self.object_points, self.image_points)
        cv2.imshow('Undistorted Test Image', undistorted_image)
        self.assertEqual(undistorted_image.shape, self.test_image.shape)

    def tearDown(self):
        del self.object_points
        del self.image_points
        del self.test_image
        del self.calibration_dict

        #saved_img_list = glob.glob(self.config['calibration_pattern_image_directory'] + '/*.jpg')
        #if len(saved_img_list) > 0:
        #    [os.remove(path) for path in saved_img_list]

        del self.config


if __name__ == '__main__':
    unittest.main()
