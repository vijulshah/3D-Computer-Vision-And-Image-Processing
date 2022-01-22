
# 3D Computer Vision

## 1. Camera Callibration
I downloaded 10 chess board images on my tablet and captured them using my laptop's webcam (see *"images/input"* folder).<br>
Then I used the **corner detection** on each image to **calibrate** and find:
* K - Camera Matrix
* R - Rotation Matrix
* T - Translation Vector
* Distortion Parameters
* Detected Corners in 3D Space
You can see the corners detected images in *"images/output"* folder.
Lastly, stored these above Parameters in the file *"camera_calibrated.mat"*

## 2. Homography
Here I have taken 2 photos of a garden *(location: Fredriksdal in Helsingborg, Sweden)*.<br> 
Both images are of same area, taken from different angles. Here, I am applying **feature matching** using Brute Furce and using **RANSAC** for outlier rejection. <br>
Lastly, I am wrapping / transforming the perspective of the image after the features are matched.<br>
Hence, first **compare the template_1.jpg and template_2.jpg** in the *images/input* folder.<br>
Now, find the result image **(FeaturesMatched.jpg)** after feature matching, outlier rejection and perspective transformation.<br>
**This result image can be seen as template_1.jpg which is perspectively changed and is now matching & aligned with template_2.jpg**

<br><br>
- - - -
# Image Processing

## 1. Object Detection

## 2. Parking Space Detection

## 3. Pose Estimator