import unittest
import numpy as np
import lane_finding as lf


class TestLaneFinding(unittest.TestCase):

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


if __name__ == '__main__':
    unittest.main()
