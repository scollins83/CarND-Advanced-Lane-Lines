import numpy as np


class LaneSpecification:

    def __init__(self, lane_width, left_fit, left_fit_x, leftx,
                 left_line, right_fit, right_fit_x, rightx,
                 right_line, middle_marker, frame_index,
                 ym_per_pix, xm_per_pix, res_yvals):
        self.used = False
        self.left_usable = False
        self.right_usable = False
        self.all_usable = False
        self.lane_width = lane_width
        self.left_fit = left_fit
        self.left_fit_x = left_fit_x
        self.leftx = leftx
        self.left_line = left_line
        self.right_fit = right_fit
        self.right_fit_x = right_fit_x
        self.rightx = rightx
        self.right_line = right_line
        self.middle_marker = middle_marker
        self.frame_index = frame_index
        self.ym_per_pix = ym_per_pix
        self.xm_per_pix = xm_per_pix
        self.res_yvals = res_yvals
        self.polynomial_width = self.right_fit[2] - self.left_fit[2]
        self.width_ok = False
        self.prev_width_ok = False
        self.two_lane_curvature_ok = False
        self.left_line_replaced = False

    def determine_usability(self, prev_lane_spec, tolerance=.85, prev_width_tolerance=.95):
        """

        :param lane_spec: Preceding line spec.
        :return:
        """

        # Check polynomial distance for lane width
        if (self.polynomial_width >= (self.lane_width*tolerance)) and (self.polynomial_width <=
                                                                       (self.lane_width*(1 + (1 - tolerance)))):
            self.width_ok = True
        # Check congruency width of previous lane_spec
        if (self.polynomial_width >= (prev_lane_spec.polynomial_width * prev_width_tolerance)) and (self.polynomial_width <=
                                                                                         (prev_lane_spec.polynomial_width*
                                                                                          (1 + (1 - prev_width_tolerance)))):
            self.prev_width_ok = True

        # Do the comparison for the 'all' settings
        if self.width_ok and self.prev_width_ok:
            #and self.prev_width_ok and self.two_lane_curvature_ok:
            self.all_usable = True
            # Left lane is still having issues with shadows, so replace just it if it deviates too far from the previous frame.
            # NOTE: If this doesn't work, try by max pixels in x
            if (np.argmax(self.left_fit_x)*(1 - tolerance)) > np.argmax(prev_lane_spec.left_fit_x):
                self.left_fit_x = prev_lane_spec.left_fit_x
                self.leftx = prev_lane_spec.leftx
                self.left_fit = prev_lane_spec.left_fit
                self.left_line = prev_lane_spec.left_line
                self.middle_marker = prev_lane_spec.middle_marker
                self.polynomial_width = self.right_fit[2] - self.left_fit[2]
                self.ym_per_pix = prev_lane_spec.ym_per_pix
                self.xm_per_pix = prev_lane_spec.xm_per_pix
                self.res_yvals = prev_lane_spec.res_yvals
                self.left_line_replaced = True
                print('Left line replaced.')


    def use_lane(self):
        self.used = True
