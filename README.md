## Self Driving Car Nano Degree

## Advanced Lane Finding

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a threshold binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/CamCalib.png "Camera calibration"
[image2]: ./output_images/undistoredChecker.png "Undistortion checker "
[image4]: ./output_images/prespectiveTransformed.png "Warp Example"
[image3]: ./output_images/undistored.png "Undistortion Road"
[image5]: ./output_images/L_channel_out.png "L channel"
[image6]: ./output_images/S_channel_out.png "S channel"
[image7]: ./output_images/H_channel_out.png "H channel"
[image8]: ./output_images/Combined_out.png "Combined Output"
[image9]: ./output_images/test_images_combined_out.png "Test Images Combined Output"
[image10]: ./output_images/Sliding_Window_Out.png "Sliding Window Output"
[image11]: ./output_images/test_images_Sliding_Window_Out.png "Test Images Sliding Window Output"
[image12]: ./output_images/lane_Overlay.png "Lane Overlay Output"
[image13]: ./output_images/text_Overlay.png "text Overlay Output"
[image14]: ./output_images/test_images_Final_Out.png "Test Images Final Output"
[video1]: ./project_video_output.gif "Video"
[video2]: ./challenge_video_output.gif "Video"


## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the second code cell of the IPython notebook located in "project.ipynb".  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `actualPoints` is just a replicated array of coordinates, and `actualPointsList` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `checkerPointsList` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `actualPointsList` and `checkerPointsList` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

*Note: Some of the chessboard images don't appear because `findChessboardCorners` was unable to detect the desired number of internal corners.*

The best effect of Undistortion can be seen in the following image

![alt text][image2]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image3]


#### 2. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `prespectivetransformer()`, which appears in third code cell in the file `project.ipynb`.  The `prespectivetransformer()` function takes as inputs an image (`img`).  I chose the hardcode the source and destination points in the following manner:

```python
 h,w = img.shape[:2]
        src = np.float32([(575,464),
                  (707,464), 
                  (258,682), 
                  (1049,682)])
        dst = np.float32([(450,0),
                  (w-450,0),
                  (450,h),
                  (w-450,h)])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 575,464       | 450,0         | 
| 707,464       | w-450,0       |
| 258,682       | 450,h         |
| 1049,682      | w-450,h       |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 3. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I have explored several combinations of the sobel and threshold gradiants. I finalized upon using HLS color space.

To extract white lines I have used `L channel` with high thresholds

![alt text][image5]

To extract white lines I have used combination of H and S channel. The H channel is tuned to extract yellow lines and S channel shall provide lane Lines which are under shadow.

![alt text][image6]

![alt text][image7]

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines # through # in `another_file.py`).  Here's an example of my output for this step.  (note: this is not actually from one of the test images)

![alt text][image8]

From Step 1 to 4, I have combined the pipeline under level1 function, which can be found in fourth code cell of file `project.ipynb` as the function name `level1_Process`. Following are the output for test images, processed using `level1_Process` pipeline

![alt text][image9]


#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

The function `find_lane_pixels` and `use_Prev_Fitting` are used to identify the lane pixels and fit a second order equation.
The steps are as follows:
    1. Find the Histogram of the bottom half of image.
    2. Identify the peaks (one each on left and right).
    3. Consider those peaks as base and iterate the windows, while iterating re-adjust the center of window according to the pixel density.
    4. From the identified pixel density centroids, generate a second degree polynomial.
    
The following image shows the output from sliding window method:

![alt text][image10]

If there is a lane equations are identified in the previous frame, then the algorithm shall look for the activated pixels only in the region in +-margin region of the last fitted equation.


Following are the outputs of the Sliding image for test samples:

![alt text][image11]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in function `get_Curv_latDist` which can be found in fourth code cell of file `project.ipynb`, this function calculates both left and right curvature and also the Lateral Offset from center.

Following are the calculations for Curvature:
    ```
       curve_radius = ((1 + (2*fit[0]*y_0*y_meters_per_pixel + fit[1])**2)**1.5) / np.absolute(2*fit[0])
    ```
The fit in the above example is the second degree polynomial equation. I have calculated the curvature at the bottom most part of image which is equal to length  of image.Hence `y_0=image.shape[0]`.

For calculation of lateral offset, I have first calculated the x value for `y=image.shape[0]` for both left and right fitted equations.
These `x` values indicate the intercept of the equations on x axis, now the lane center can be estimated as average of the `left x intercept` and `right x intercept`.

Considering vehicle is at the center of the image or Camera is mounted at the center of vehicle, we can calculate lateral offset by subtracting the lane center calculated by intercepts and the image center(`(image.shape[1]/2)`).

Lateral Offset=`((left x intercept + right x intercept)/2) - (image.shape[1]/2)`


#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I have implemented the final pipeline which can be found in fifth code cell of file `project.ipynb` as the function name `lane_Overlay`.  Here is an example of my result on a test image:

![alt text][image12]

Following is the sample after Overlay of text string on the output image.

![alt text][image13]


Following are the outputs of the `finalpipline` for test samples:

![alt text][image14]


---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's an output of the pipeline

##### project_video_output

![alt text][video1]


##### challenge_video_output

![alt text][video2]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Following are the Problems and Limitations of the current implementation.
    1. One of the major issues was the identification of lanes using Color and thresholding, the problem here is since I knew the conditions in project video I did my best to tune the thresholds, but the greater challenge would be to identify a generic solution which can be used for any lighting conditions. I look forward to implement adaptive thresholds for the level1 processing. Also I haven’t used sobel operator anywhere, however in the next version I would like to add sobel based thersholding to make lane identification more robust.
    2. There is were also false identification of lane lines, other than the Ego-Lane. To eliminate this I did Region of Interest masking, but the major disadvantage using this is that if the lane has steep curve the mask would just discard most of the curvature. One way to get around is to implement an algo which shall identify all the lane markings in Camera FOV and segregate them as needed.
    3.Sliding Window does a good job in following the lane lines as long as there is less noise in the margin parameter of the algorithm.
    4. Also I have implemented averaging of good fits until 10 best fits.
    5. I have implemented false or highly deviating fits rejection by checking the difference in x intercepts of fit which should be close to the lane width in pixels. Also there is second level filtering where I check the difference in the best fit and current fit and discard the fit if it exceeds the limits. But this implementation is very rudimentary and can easily be deceived.

Future Goals:
    1. Implement adaptive thresholding based on the image characteristics.
    2. Implement a robust fit rejection algorithm to intelligently identify the false fits and discard them.
    3. Multiple lane identification and tracking can also be very helpful in order to understand the environment.
    
I have also tried the harder_challenge videos following are the links for the same:

[harder challenge video output](./harder_challenge_video_output.mp4)
