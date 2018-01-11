## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[original_calibration_image]: ./test/unit_tests/test_image.png "Original calibration image"
[undistorted_calibration_image]: ./test/unit_tests/und_image.png "Undistorted"
[undistorted]: ./tuning_images/run4_3_undistorted.jpg "Undistorted road"
[original]: ./test_images/tracked_3.jpg "Original road"
[binary]: ./tuning_images/run4_3_binary.jpg "Binary Example"
[warped]: ./tuning_images/run4_3_warped.jpg "Warp Example"
[warp_boxes]: ./test_images/tracked_3_overlay.jpg "Warp Lines Highlighted"
[lines_only]: ./test_images/tracked_3_road.jpg "Road lines only"
[lines_on_road]: ./test_images/tracked_3_road_round3.jpg "Lines on Road"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

Please see the rest of this file. 

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is located in the script [lane_finding_calibration.py](../lane_finding_calibration.py)
That particular script consists of functions, and the 'driver' is run in order under the default main in line 204. I set up my scripts to use configuration files for parameters as I find these to be extremely useful in being able to iterate quickly and they also make it easy to maintain a record of sorts of which parameters I used. My unit testing calibration file was sufficient to train this model, and that configuration file is located in test/unit_tests/test_calibration_configuration.json.

My lane_finding_calibration.py driver script calculates the object and image points in a single function, which wraps together several other functions in the script. That wrapper function is 'get_calibration_object_and_image_points', which begins at line 133. The first call is to create_object_points (line 37), which creates an array of zeros, and then reshapes that array with the 'modify_object_points' function starting in line 49. The wrapper then gets the list of images (line 65) and iterates through them, first converting them to grayscale (function starts on line 83) and then finding the chessboard corners (function starts on line 93). If corners are found, the object points and corners are retained, the new corner points are drawn onto images and saved (function starts line 112), and then the retained lists of object points and image points are returned. From there, a calibration image is read in (line 222), the size noted (line 223), and a calibration dictionary is generated from the function 'calibrate_camera', which starts on line 192, and this dictionary contains the camera matrix and the distortion coefficients. Lastly, that calibration dictionary is saved (line 225), and the driver exits. 

The wrapper function start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][original_calibration_image]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][undistorted_calibration_image]

For the remaineder of the code, I switched to the image_processing.py script, and again, the driver is in the default main method starting at line 192. I paused and coded along during the walkthrough video for this project a lot which is where a lot of the code comes from, in addition to the class content functions, but did refactor a few things and then tuned the input parameters. 

In this case, I loaded the saved camera calibration dictionary, and applied the cv2 'undistort' function in line 214, and saved the output in order to be able to tune the code later and provide images for this writeup. 

Original image in following processing examples:
![alt text][original]  

Undistorted:  
![alt text][undistorted]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used both sobel thresholds (function begins on line 74) and color thresholds (function begins on line 154) to generate a binary image (thresholding steps at lines 220 through 225 in `image_processing.py`).  Here's an example of my output for this step.

![alt text][binary]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform occurs in lines 232 through 249 in the file `image_processing.py`. I somewhat programmatically determined a trapezoid to use for the source rectangle on the image to warp, in lines 233 through 236. While this didn't yield as potentially perfect a result as hardcoding the points likely would have, as long as the trapezoid din't appear to be too far off of any one given road, this approach seemed more conducive to allowing the code to adapt to several different types of roads which the car may encounter. 

The destination was selected using an offset percentage, and then was subtracted from the image width where necessary in lines 242 through 245. 

This resulted in the following source and destination points, programmatically:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 460      | 320, 0        | 
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 695, 460      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image, although I did not retain these images from preprocessing.

However, I did save out the warped images themselves and show and example here, and used the appearance of the lane lines in these transforms and the appearance of noise to tune the offset and trapezoid measurement numbers to maximize bold lines and the least amount of noise in the transformed pictures in my sample images:

![alt text][warped]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

I used the one-dimensional convolution method noted in the class content and walkthrough video to find my lane lines, as it was a technique I hadn't tried before (I used histograms in past work with audio for event detection and timbre analysis). This code starts in image_processing.py line 255 through 303, and the Tracker class (used in the place of the suggested 'line' class for storing line attributes) in tracker.py. I tuned the height and width of the convolution window to find the lines, and then used a tracker object to return the curve centers (image_preprocessing.py, line 258). I then used the curve centers to find the window centroids of my warped image (image_preprocessing.py line 264, and tracker.py line 26 through 60). With those found, the window mask was used in order to be able to draw those areas on the image, and that's shown in image_processing.py lines 272 through 283, and the boxed lines were drawn onto the warped image. I didn't consistently use this view in tuning, but it was a helpful visualization to show how the convolution window height and width, along with other noise in the warp image, contributed to the video outcomes. 

![alt text][warped_boxes]

In order to actually get bona-fide lane lines from those convolutions, I used the numpy polyfit function to fit the left and right lanes to a 2nd degree polynomial (I did also explore what other orders of polynomials would do during tuning, but those results obviously weren't great). 

After finding those lines, clear lines could be obtained.  
![alt text][lines_only]  
  
From there, I was able to draw the lines back on the road
![alt text][lines_on_road]  
  
#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
