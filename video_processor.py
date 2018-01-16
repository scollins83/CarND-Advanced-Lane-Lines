from moviepy.editor import VideoFileClip
from IPython.display import HTML
import numpy as np
import cv2
import pickle
import glob
from tracker import Tracker
import os
import lane_finding_calibration as lfc
import logging
import sys
import time
import lane_specification as ls


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def hls_select(img, thresh=(0, 255)):
    """
    Thresholds the S-channel of HLS color space.
    :param img: Image array.
    :param thresh: Threshold tuple (min, max)
    :return: Binary output.
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
    :param img: Image array
    :param orient: Orientation for sobel thresholding
    :param thresh_min: Threshold minimum
    :param thresh_max: Threshold maximum
    :return: Binary image result
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


def color_threshold(image, sthresh=(0, 255), vthresh=(0, 255), lthresh=(0, 255)):
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

    l_channel = hls[:, :, 1]
    l_binary = np.zeros_like(l_channel)
    l_binary[(l_channel >= lthresh[0]) & (l_channel <= lthresh[1])] = 1

    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    v_channel = hsv[:, :, 2]
    v_binary = np.zeros_like(v_channel)
    v_binary[(v_channel >= vthresh[0]) & (v_channel <= vthresh[1])] = 1

    output = np.zeros_like(s_channel)
    output[(s_binary == 1) & (v_binary == 1) & (l_binary == 1)] = 1
    return output

def window_mask(width, height, img, center, level):
    """
    Mask for drawing window areas for convolutional windows.
    :param width: Window width
    :param height: Window height
    :param img: Image array
    :param center: Window centroid level center
    :param level: Window centroid level
    :return: Masked window
    """
    output = np.zeros_like(img)
    output[int(img.shape[0] - (level + 1) * height):int(img.shape[0] - level * height),
    max(0, int(center - width)):min(int(center + width), img.shape[1])] = 1
    return output


def process_image(img):
    """
    Process the image for each video frame.
    :param img: Image array.
    :return: Processed image.
    """

    # Assert that at least a 3-channel image was passed to the function.
    assert img.shape[-1] == 3

    # Increment the global frame_counter
    global frame_counter
    #global last_used_frame_index
    #global lanes
    global last_lane_used

    current_frame = frame_counter

    # Undistort the image.
    img = cv2.undistort(img, matrix, dist, None, matrix)

    # Process image and generate binary pictures
    preprocessed_image = np.zeros_like(img[:, :, 0])
    gradx = abs_sobel_thresh(img, orient='x', thresh_min=config['sobel_x_min'], thresh_max=config['sobel_x_max'])
    grady = abs_sobel_thresh(img, orient='y', thresh_min=config['sobel_y_min'], thresh_max=config['sobel_y_max'])
    c_binary = color_threshold(img, sthresh=(config['color_s_thresh_min'], config['color_s_thresh_max']),
                               vthresh=(config['color_v_thresh_min'], config['color_v_thresh_max']),
                               lthresh = (config['color_l_thresh_min'], config['color_l_thresh_max']))

    preprocessed_image[((gradx == 1) & (grady == 1) | (c_binary == 1))] = 255
    # Lots of experimentation to how to get the best binary images

    # Set up Perspective Transform Area
    img_size = (img.shape[1], img.shape[0])

    # Set up source and destination coordinates for transform.
    src = np.float32([[config['src_0_x'], config['src_0_y']],
                      [config['src_1_x'], config['src_1_y']],
                      [config['src_2_x'], config['src_2_y']],
                      [config['src_3_x'], config['src_3_y']]])
    dst = np.float32([[config['dst_0_x'], config['dst_0_y']],
                      [config['dst_1_x'], config['dst_1_y']],
                      [config['dst_2_x'], config['dst_2_y']],
                      [config['dst_3_x'], config['dst_3_y']]])

    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(preprocessed_image, M, img_size, flags=cv2.INTER_LINEAR)

    # Set up the overall Tracker class object to track lane line data

    curve_centers = Tracker(window_width=config['conv_window_width'], window_height=config['conv_window_height'],
                            margin=config['tracker_margin'],
                            ym=config['tracker_ym_numerator'] / config['tracker_ym_denominator'],
                            xm=config['tracker_xm_numerator'] / config['tracker_xm_denominator'],
                            smooth_factor=config['tracker_smoothing_factor'])

    # Finds center points of windows to use to draw lane lines.
    window_centroids = curve_centers.find_window_centroids(warped)
    #logger.info("Number of window centroids: " + str(len(window_centroids)))

    l_points = np.zeros_like(warped)
    r_points = np.zeros_like(warped)

    leftx = []
    rightx = []

    for level in range(0, len(window_centroids)):
        # Level == Number of boxes to stack to make a line
        # Window mask is function to draw window areas
        # Add center value found in frame to the list of the lane points per left, and right lines
        leftx.append(window_centroids[level][0])
        rightx.append(window_centroids[level][1])

        l_mask = window_mask(config['conv_window_width'], config['conv_window_height'],
                             warped, window_centroids[level][0], level)
        r_mask = window_mask(config['conv_window_width'], config['conv_window_height'],
                             warped, window_centroids[level][1], level)

        # Add graphic points from window mask here to total pixels found
        l_points[(l_points == 255) | ((l_mask == 1))] = 255
        r_points[(r_points == 255) | ((r_mask == 1))] = 255

        # If level == 0, and none found, start looking for an escape route.

    # Draw the results
    template = np.array(r_points + l_points, np.uint8)
    zero_channel = np.zeros_like(template)
    template = np.array(cv2.merge((zero_channel, template, zero_channel)), np.uint8)  # Making windows green
    warpage = np.array(cv2.merge((warped, warped, warped)), np.uint8)

    result = cv2.addWeighted(warpage, 1, template, 0.5,
                             0.0)  # Overlay the original road img with window results; not showing up.

    # Fit lane boundaries to left, right, center positions found
    yvals = range(0, warped.shape[0])
    res_yvals = np.arange(warped.shape[0] - (config['conv_window_height'] / 2), 0, - config['conv_window_height'])

    left_fit = np.polyfit(res_yvals, leftx, 2)
    left_fitx = left_fit[0] * yvals * yvals + left_fit[1] * yvals + left_fit[2]
    left_fitx = np.array(left_fitx, np.int32)

    right_fit = np.polyfit(res_yvals, rightx, 2)
    right_fitx = right_fit[0] * yvals * yvals + right_fit[1] * yvals + right_fit[2]
    right_fitx = np.array(right_fitx, np.int32)

    left_lane = np.array(list(
        zip(np.concatenate((left_fitx - config['conv_window_width'] / 2,
                            left_fitx[::-1] + config['conv_window_width'] / 2), axis=0),
            np.concatenate((yvals, yvals[::-1]), axis=0))), np.int32)
    right_lane = np.array(list(
        zip(np.concatenate((right_fitx - config['conv_window_width'] / 2,
                            right_fitx[::-1] + config['conv_window_width'] / 2), axis=0),
            np.concatenate((yvals, yvals[::-1]), axis=0))), np.int32)
    middle_marker = np.array(list(
        zip(np.concatenate((left_fitx + config['conv_window_width'] / 2,
                            right_fitx[::-1] - config['conv_window_width'] / 2), axis=0),
            np.concatenate((yvals, yvals[::-1]), axis=0))), np.int32)

    ym_per_pix = curve_centers.ym_per_pix
    xm_per_pix = curve_centers.xm_per_pix

    # Turn the lane objects to the list of the lane lines
    lane = ls.LaneSpecification(config['tracker_xm_denominator'],
                                left_fit,
                                left_fitx,
                                leftx,
                                left_lane,
                                right_fit,
                                right_fitx,
                                rightx,
                                right_lane,
                                middle_marker,
                                current_frame,
                                ym_per_pix,
                                xm_per_pix,
                                res_yvals)

    if frame_counter == 0:
        lane.use_lane()
        last_lane_used = lane

    else:
        lane.determine_usability(last_lane_used)
        if lane.all_usable:
            lane.use_lane()
            last_lane_used = lane
            if lane.left_line_replaced:
                left_lane = lane.left_line
                middle_marker = lane.middle_marker
                leftx = lane.leftx
                ym_per_pix = lane.ym_per_pix
                xm_per_pix = lane.xm_per_pix
                res_yvals = lane.res_yvals
                left_fitx = lane.left_fit_x

        else:
            left_lane = last_lane_used.left_line
            right_lane = last_lane_used.right_line
            middle_marker = last_lane_used.middle_marker
            leftx = last_lane_used.leftx
            rightx = last_lane_used.rightx
            ym_per_pix = last_lane_used.ym_per_pix
            xm_per_pix = last_lane_used.xm_per_pix
            res_yvals = last_lane_used.res_yvals
            right_fitx = lane.right_fit_x
            left_fitx = lane.left_fit_x


    road = np.zeros_like(img)
    road_bkg = np.zeros_like(img)
    cv2.fillPoly(road, [left_lane], color=[255, 0, 0])
    cv2.fillPoly(road, [right_lane], color=[0, 0, 255])
    cv2.fillPoly(road, [middle_marker], color=[0, 255, 0])
    cv2.fillPoly(road_bkg, [left_lane], color=[255, 255, 255])
    cv2.fillPoly(road_bkg, [right_lane], color=[255, 255, 255])

    road_warped = cv2.warpPerspective(road, Minv, img_size, flags=cv2.INTER_LINEAR)
    road_warped_bkg = cv2.warpPerspective(road_bkg, Minv, img_size, flags=cv2.INTER_LINEAR)

    base = cv2.addWeighted(img, 1.0, road_warped_bkg, -1.0, 0.0)
    result = cv2.addWeighted(base, 1.0, road_warped, 1.0, 0.0)


    left_curve_fit_cr = np.polyfit(np.array(res_yvals, np.float32) * ym_per_pix,
                                   np.array(leftx, np.float32) * xm_per_pix, 2)
    left_curverad = ((1 + (
            2 * left_curve_fit_cr[0] * yvals[-1] * ym_per_pix + left_curve_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * left_curve_fit_cr[0])

    right_curve_fit_cr = np.polyfit(np.array(res_yvals, np.float32) * ym_per_pix,
                                    np.array(rightx, np.float32) * xm_per_pix, 2)
    right_curverad = ((1 + (
            2 * right_curve_fit_cr[0] * yvals[-1] * ym_per_pix + right_curve_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * right_curve_fit_cr[0])

    camera_center = (left_fitx[-1] + right_fitx[-1]) / 2
    center_diff = (camera_center - warped.shape[1] / 2) * xm_per_pix
    side_pos = 'left'
    if center_diff <= 0:
        side_pos = 'right'

    cv2.putText(result, 'Left radius of curvature = ' + str(round(left_curverad, 3)) + '(m)', (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(result, 'Right radius of curvature = ' + str(round(right_curverad, 3)) + '(m)', (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(result, "Vehicle is " + str(abs(round(center_diff, 3))) + 'm ' + side_pos + ' of center', (50, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(result, 'Frame Number: ' + str(current_frame), (50, 200), cv2.FONT_HERSHEY_SIMPLEX,
                1, (255, 255, 255), 2)

    frame_counter = frame_counter + 1
    return result


if __name__ == "__main__":

    # Load the necessary parameters
    args = lfc.parse_args(sys.argv[1:])
    config = lfc.load_config(args['config'])

    # Read in saved object points and image points
    calibration = pickle.load(open(config['calibration_pickle'], 'rb'))
    matrix = calibration['camera_matrix']
    dist = calibration['distortion_coeffs']

    output_video = config['output_video']
    input_video = config['input_video']

    frame_counter = 0
    #last_used_frame_index = 0
    #lanes = []

    original_video_clip = VideoFileClip(input_video)
    processed_video_clip = original_video_clip.fl_image(process_image)

    # Save the processed video file.
    processed_video_clip.write_videofile(output_video, audio=False)

    # Print the framecount
    logger.info("Frames processed: " + str(frame_counter))

    sys.exit(0)
