import cv2
import lane_finding_calibration as lfc
import os
import sys
import logging
import numpy as np
import pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def undistort_image(image, object_points, image_points):
    """

    :param image:
    :param object_points:
    :param image_points:
    :return:
    """
    # Convert image to grayscale
    gray_scale_image = lfc.convert_color_image_to_grayscale(image)

    # Calibrate the camera
    calibration_dict = lfc.calibrate_camera(object_points, image_points, gray_scale_image.shape[::-1])

    # Return the undistorted image
    return cv2.undistort(image, calibration_dict['camera_matrix'],
                         calibration_dict['distortion_coeffs'],
                         None,
                         calibration_dict['camera_matrix'])


def unwarp_corners(image, corners_width, corners_height, pattern_corners, camera_matrix, dist_coefficients):
    """

    :param image:
    :param corners_width:
    :param corners_height:
    :param pattern_corners:
    :param camera_matrix:
    :param dist_coefficients:
    :return:
    """
    undistorted_image = cv2.undistort(image, camera_matrix, dist_coefficients,
                                      None, camera_matrix)

    gray_undistorted = cv2.cvtColor()

    if pattern_corners == "None":
        input_corners = None
    else:
        input_corners = pattern_corners

    ret, corners = cv2.findChessboardCorners(undistorted_image, (corners_width, corners_height), input_corners)

    if ret == True:
        offset = 100
        image_size = (gray_undistorted.shape[1], gray_undistorted.shape[0])
        source_points = np.float32([corners[0], corners[corners_width - 1],
                                    corners[-1], corners[-corners_width]])
        destination_points = np.float32([[offset, offset],
                                         [image_size[0] - offset,
                                         offset],
                                        [image_size[0] - offset,
                                         image_size[1] - offset],
                                        [offset, image_size[1] - offset]])

        matrix = cv2.getPerspectiveTransform(source_points, destination_points)
        warped_image = cv2.warpPerspective(undistorted_image, matrix, image_size)

        return warped_image, matrix


def hls_select(img, thresh=(0, 255)):
    """
    thresholds the S-channel of HLS
    :param img:
    :param thresh:
    :return:

    hls_binary = hls_select(image, thresh=(90, 255)
    """
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]
    binary_output = np.zeros_like(s_channel)
    binary_output[(s_channel > thresh[0]) & (s_channel <= thresh[1])] = 1

    return binary_output


def abs_sobel_thresh(img, orient='x', thresh_min=0, thresh_max=255):
    """
    Takes an image, gradient orientation,
    # and threshold min / max values.
    :param img:
    :param orient:
    :param thresh_min:
    :param thresh_max:
    :return:
    """
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

    # Return the result
    return binary_output


def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    """
    Returns the magnitude of the gradient # for a given sobel kernel size and threshold values

    :param img:
    :param sobel_kernel:
    :param mag_thresh:
    :return:
    """
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255
    gradmag = (gradmag/scale_factor).astype(np.uint8)
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

    # Return the binary image
    return binary_output



def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    """
    Thresholds an image for a given range and Sobel kernel
    :param img:
    :param sobel_kernel:
    :param thresh:
    :return:
    """
    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction,
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output =  np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

    # Return the binary image
    return binary_output

# Combination for HSV and HLS color_threshold
def color_threshold(image, sthresh=(0, 255), vthresh=(0, 255)):
    """
    Combines HSV and HLS
    :param image:
    :param sthresh:
    :param vthresh:
    :return:
    """
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    s_channel = hls[:, :, 2]
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= sthresh[0]) & (s_channel <= sthresh[1])] = 1

    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    v_channel = hsv[:, :, 2]
    v_binary = np.zeros_like(v_channel)
    v_binary[(v_channel >= vthresh[0]) & (v_channel <= vthresh[1])] = 1

    output = np.zeros_like(s_channel)
    output[(s_binary == 1) & (v_binary == 1)] = 1
    return output


def window_mask(width, height, img_ref, center, level):
    """

    :param width:
    :param height:
    :param img_ref:
    :param center:
    :param level:
    :return:
    """
    output = np.zeros_like(img_ref)
    return output




if __name__ == "__main__":

    # Set TensorFlow logging so it isn't so verbose.
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    # Load the necessary parameters
    args = lfc.parse_args(sys.argv[1:])
    config = lfc.load_config(args['config'])

    # Read in saved object points and image points
    calibration = pickle.load(open(config['calibration_pickle'], 'rb'))
    matrix = calibration['camera_matrix']
    dist = calibration['distortion_coeffs']
    logger.info(matrix.shape)
    logger.info(dist.shape)

    image_paths = lfc.get_image_file_paths(config['image_list'])

    for index, filename in enumerate(image_paths):
        # Read the image
        img = cv2.imread(filename)
        # Undistort
        img = cv2.undistort(img, matrix, dist, None, matrix)

        write_name = config['undistorted_save_pattern'] + str(index) + '.jpg'
        cv2.imwrite(write_name, img)

        # Process image and generate binary pixel
        preprocessed_image = np.zeros_like(img[:, :, 0])
        gradx = abs_sobel_thresh(img, orient='x', thresh_min=12, thresh_max=255)
        grady = abs_sobel_thresh(img, orient='y', thresh_min=25, thresh_max=255)
        c_binary = color_threshold(img, sthresh=(100, 255), vthresh=(50, 255))
        preprocessed_image[((gradx == 1) & (grady == 1) | (c_binary == 1))] = 255
        # Lots of experimentation to how to get the best binary images

        write_name = config['undistorted_save_pattern'] + str(index) + '_binary.jpg'
        cv2.imwrite(write_name, preprocessed_image)


    sys.exit(0)
