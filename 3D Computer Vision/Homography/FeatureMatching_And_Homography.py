# Brute-Force Matching with ORB Descriptors
import numpy as np
import cv2

im1 = cv2.imread('images/input/feature_matching_template_1.jpg') # Image that needs to be registered.
im2 = cv2.imread('images/input/feature_matching_template_2.jpg') # train Image

img1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

# Initiate ORB detector
orb = cv2.ORB_create(50)  # Registration works with at least 50 points

# find the keypoints and descriptors with orb
kp1, des1 = orb.detectAndCompute(img1, None) # kp1 --> list of keypoints
kp2, des2 = orb.detectAndCompute(img2, None)

# Brute-Force matcher takes the descriptor of one feature in first set and is matched with all other features in second set using some distance calculation.
# create Matcher object

matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)

# Match descriptors.
matches = matcher.match(des1, des2, None)  # Creates a list of all matches, just like keypoints

# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)

# https://docs.opencv.org/3.0-beta/modules/features2d/doc/drawing_function_of_keypoints_and_matches.html
# Draw first 100 matches.
img3 = cv2.drawMatches(im1,kp1, im2, kp2, matches[:100], None)

cv2.imshow("Matches image", cv2.resize(img3,(1024, 1024)))
cv2.imwrite("images/output/matches.jpg", img3)
cv2.waitKey(0)

# Now let us use these key points to register two images. 
# Can be used for distortion correction or alignment
# For this task we will use homography. 
# https://docs.opencv.org/3.4.1/d9/dab/tutorial_homography.html

# Extract location of good matches.
# For this we will use RANSAC (RANdom SAmple Consensus).
# It is an outlier rejection method for keypoints.
# http://eric-yuan.me/ransac/
# RANSAC needs all key points indexed, first set indexed to queryIdx
# Second set to #trainIdx. 

points1 = np.zeros((len(matches), 2), dtype=np.float32)  # Prints empty array of size equal to (matches, 2)
points2 = np.zeros((len(matches), 2), dtype=np.float32)

for i, match in enumerate(matches):
   points1[i, :] = kp1[match.queryIdx].pt # gives index of the descriptor in the list of query descriptors
   points2[i, :] = kp2[match.trainIdx].pt # gives index of the descriptor in the list of train descriptors

# Now we have all good keypoints so we are ready for homography.   
# Find homography
# https://en.wikipedia.org/wiki/Homography_(computer_vision)
  
h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
 
# Use homography
height, width, channels = im2.shape
im1Reg = cv2.warpPerspective(im1, h, (width, height)) # Applies a perspective transformation to an image.
   
print("Estimated homography : \n",  h)
cv2.imshow("Features Matched Image", cv2.resize(im1Reg,(1024, 1024)))
cv2.imwrite("images/output/FeaturesMatched.jpg", im1Reg)
cv2.waitKey()