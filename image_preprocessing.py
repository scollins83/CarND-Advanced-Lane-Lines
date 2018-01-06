import cv2
import lane_finding_calibration as lfc

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
    return_val, camera_matrix, distortion_coeffs, rotation_vecs, \
    translation_vecs = cv2.calibrateCamera(object_points, image_points, gray_scale_image.shape[::-1], None, None)

    # Return the undistorted image
    return cv2.undistort(image, camera_matrix, distortion_coeffs, None, camera_matrix)

