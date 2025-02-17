# **Advanced Lane Finding Project**

### Sara Collins  
  
  
##### To generate video: 
1) Fill out desired parameters in `local_configuration.json`, or else copy that file and supply other parameters.
2) Run `python video_processor.py -c local_configuration.json`

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
[undistorted]: ./output_images/example_2_undistorted.jpg "Undistorted road"
[original]: ./test_images/test4.jpg "Original road"
[binary]: ./output_images/example_2_binary.jpg "Binary Example"
[warped]: ./output_images/example_2_warped.jpg "Warp Example"
[warp_boxes]: ./output_images/example_2_overlay.jpg "Warp Lines Highlighted"
[lines_only]: ./output_images/example_2_road.jpg "Road lines only"
[lines_on_road]: ./output_images/example_2_lines_on_road.jpg "Lines on Road"
[image_on_road]: ./output_images/example_2_road_round4.jpg "Output"
[video]: ./project_output_video.mp4 "Video"

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

For the remainder of the code, I switched to the `video_processor.py` script, and again, the driver is in the default main method starting at line 372, and the main image processing is wrapped in a function that begins on line 157. I paused and coded along during the walkthrough video for this project a lot which is where a lot of the code comes from, in addition to the class content functions, but did refactor quite a few things during tuning and then tuned the input parameters. 

In this case, I loaded the saved camera calibration dictionary, and applied the cv2 'undistort' function in line 176, and saved the output in order to be able to tune the code later and provide images for this writeup. 

Original image in following processing examples:
![alt text][original]  

Undistorted:  
![alt text][undistorted]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used both absolute sobel thresholds (function begins on lines 33-61) and color thresholds (function  on lines 114 - 139) to generate a binary image (thresholding steps at lines 179 through 186 in `video_processor.py`).  The next image shows an example of my output for this step.
Due to having had issues with tree shadows over the left line, I built in parameters to also threshold the 'L' HLS channel for lightness as that channel was important in augmenting my images to deal with shadows 
for the Behavioral Cloning project, but quickly realized that I wanted to include the whole range of lightness, not filter any particular values. Thus, I set the parameters in `local_configuration.json` for 
'color_l_thresh' min and max to 0 and 255, respectively, to essentially disable that portion of the `color_threshold` function. Instead, I used the S channel from HLS, and the v channel from HSV, setting values by examining arrays of values in my debugger. Throughout development, I also experimented with other color channels as well, so there are several parameters and ranges available in the `local_configuration.json` file that my code didn't use, but that I wanted to save for future use. 

![alt text][binary]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform occurs in lines 122 through 130 in the file `video_processor.py`. While I initially tried to work with the somewhat programmatically determined trapezoid shape discussed in the walkthrough, I found during tuning that using hardcoded values, like what was noted in the class content worked much better for this video. 

After several rounds of trial and error, I settled on the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 550, 460      | 200, 0        | 
| 225, 700      | 300, 720      |
| 1120, 700     | 850, 720      |
| 740, 460      | 1000, 0       |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image, although unfortunately I later accidentally overwrote the directory where I kept those particular images and thus did not retain them from preprocessing.

However, I did save out the warped images themselves and show and example here, and used the appearance of the lane lines in these transforms and the appearance of noise to tune the offset and trapezoid measurement numbers to maximize bold lines and the least amount of noise around the lane lines in the transformed pictures in my sample images:

![alt text][warped]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

I used the one-dimensional convolution method noted in the class content and walkthrough video to find my lane lines, as it was a technique I hadn't tried before (I used histograms exclusively in past work with audio for event detection and timbre analysis). This code starts in video_processor.py line 208 through 274, and the Tracker class (used to store line attributes and compute window centroids for the convolution) in tracker.py. I tuned the height and width of the convolution window to find the lines, and then used a tracker object to return the curve centers (`video_processor.py`, line 208). 
I then used the curve centers to find the window centroids of my warped image (`video_processor.py` line 215, and `tracker.py` line 26 through 60). 
With those found, the window mask was used in order to be able to draw those areas on the image, and that's shown in `video_processor.py` lines 214 through 233, and the boxed lines were drawn onto the warped image. 

![alt text][warp_boxes]

In order to actually get bona-fide lane lines from those convolutions, I used the numpy polyfit function to fit the left and right lanes to a 2nd degree polynomial (I did also explore what other orders of polynomials would do during tuning, but those results obviously weren't great). 
This code is in `video_processor.py` lines 252 through 274. 

After finding those lines, clear lines could be obtained.  
![alt text][lines_only]  
  

Note also that I implemented a fairly simple smoothing mechanism using the class indicated in `lane_specification.py`. In that file, I stored each 'lane' information, including information from both right and left. From there, I coded in a couple of sanity checks to indicate whether the lane should be used in it's entirety, if it should be replaced by the last used lane, or whether only the left line and middle marker should be replaced by the preceding lane caputure's left line, middle marker, and measurements needed to calculate the radius of curvature and centeredness. The sanity checks that made the 'cut' were:
1) Checking the distance on the polynomial fit object to see if the pixel distance was within +/-15% of the noted xm_per_pixel denominator value  
2) Checking to see if the width between the polynomial fit objects were within +/-5% of the previous frames' width.  
3) If passing those, the left lane's max arg for the left_fix_x pixels to see if at least 95% of that value was greater than that of the preceding frame. If so, the left lane line, the middle marker, and the measurements needed for radius of curvature calculation was replaced with that of the preceding used frame. 
  
#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines 341 through 354 in my code in `video_processor.py`. First, the x and y meters per pixel were estimated by using 3.7 meters as an estimate for the width of the lane, and 25.5 meters estimated as the length of the used portion of the lane, as measured by estimating 3m per dashed line, and taking the pixel height of a typical straight dashed line in a warped image and dividing the image height by that number.
Those values were set in the `local_configuration.json` file, and passed to the Tracker object. I then obtained the ym_per_pixel and xm_per_pixel values from the Tracker's curve centers object, the lines were fit to the lane line boundary values found to draw the previous example picture. The curve radius was then calculated for each line. 
Initially upon my first submission, I had those yardage estimates really wrong, but now they appear to be performing more reasonably (except in spots where it is difficult to recognize the line at all, in which case it jumps around).

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines 326 through 338 in my code in `video_processor.py`- I filled in the lane with green, and then added the radius of curvature and center lane information in lines 355 - 366.  
Here is an example of my result on a test image:

![alt text][image_on_road]


---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_output_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The current smoothing I picked isn't particularly robust, and is highly dependent on the initial image, which is a failure point when checked on the challenge video. In the future, I'd change that and average things a bit more over more frames, or use some sort of moving average if possible. 
Also, I found some really good color thresholds that should have worked, but trying to optimize them to all work together was pretty daunting for not much more payoff than I was already getting with the rest of my implementation. However, I think improving that would considerably improve this implementation's robustness. Also, this model is sensitive to writing on the roads, so I'd want to do something to exclude the parts in between the lane lines. 
