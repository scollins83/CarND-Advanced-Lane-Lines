import numpy as np
import cv2


class Tracker():

    def __init__(self, window_width, window_height,
                 margin, ym = 1, xm = 1,
                 smooth_factor = 15):

        self.recent_centers = []

        self.window_width = window_width

        self.window_height = window_height

        self.margin = margin

        self.ym_per_pix = ym

        self.xm_per_pix = xm

        self.smooth_factor = smooth_factor


    def find_window_centroids(self, warped):

        window_width = self.window_width
        window_height = self.window_height
        margin = self.margin

        window_centroids = []
        window = np.ones(window_width)

        # This is where the histogram gets squashed - looks at lowest quarter
        l_sum = np.sum(warped[int(3*warped.shape[0]/4):, :int(warped.shape[1]/2)], axis=0)
        l_center = np.argmax(np.convolve(window, l_sum)) - window_width/2
        r_sum = np.sum(warped[int(3*warped.shape[0]/4):, int(warped.shape[1]/2):], axis=0)
        r_center = np.argmax(np.convolve(window, r_sum))-window_width/2+int(warped.shape[1]/2)

        window_centroids.append((l_center, r_center))

        for level in range(1, (int)(warped.shape[0]/window_height)):
            image_layer = np.sum(warped[int(warped.shape[0]-(level+1)*window_height):int(warped.shape[0]-level*window_height),:], axis=0)
            conv_signal = np.convolve(window, image_layer)

            offset = window_width/2
            l_min_index = int(max(l_center+offset-margin, 0))
            l_max_index = int(min(l_center+offset-margin, warped.shape[1]))
            l_center = np.argmax(conv_signal[l_min_index:l_max_index])+l_min_index-offset

            r_min_index = int(max(r_center+offset-margin, 0))
            r_max_index = int(min(r_center+offset+margin, warped.shape[1]))
            r_center = np.argmax(conv_signal[r_min_index:r_max_index])+r_min_index-offset

            window_centroids.append((l_center, r_center))

        self.recent_centers.append(window_centroids)

        return np.average(self.recent_centers[-self.smooth_factor:], axis=0)
    # TODO: Start back in on walkthrough video at 38:59