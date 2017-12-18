import unittest
import numpy as np
import lane_finding as lf


class MyTestCase(unittest.TestCase):

    def test_create_object_points(self):
        grid_height = 6
        grid_width = 9
        grid_depth = 3
        tester_object = np.zeros((grid_height*grid_width,grid_depth), np.float32)
        grid_object = lf.create_object_points(grid_height, grid_width, grid_depth)
        self.assertEqual(tester_object.shape, grid_object.shape)


if __name__ == '__main__':
    unittest.main()
