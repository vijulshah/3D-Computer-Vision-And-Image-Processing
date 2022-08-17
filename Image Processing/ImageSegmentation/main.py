import numpy as np
import cv2

# There are 3 features, say, R,G,B. So we need to reshape the image to an array of Mx3 size (M is number of pixels in image)
img = cv2.imread('sample.jpg')
img2 = img.reshape((-1,3))
image_to_segment = np.float32(img2)

# define criteria, number of clusters(K) and apply kmeans()
# INPUTS:
# cv.TERM_CRITERIA_EPS - stop the algorithm iteration if specified accuracy, epsilon, is reached. OR
# cv.TERM_CRITERIA_MAX_ITER - stop the algorithm after the specified number of iterations, max_iter. OR
# cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER - stop the iteration when any of the above condition is met.
# max_iter - An integer specifying maximum number of iterations.
# epsilon - Required accuracy
# attempts - Flag to specify the number of times the algorithm is executed using different initial labellings. The algorithm returns the labels that yield the best compactness. This compactness is returned as output.
# flags : This flag is used to specify how initial centers are taken. Normally two flags are used for this : cv.KMEANS_PP_CENTERS and cv.KMEANS_RANDOM_CENTERS.

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 12
attempts = 10
flags = cv2.KMEANS_RANDOM_CENTERS

compactness, labels, centers = cv2.kmeans(image_to_segment,K,None,criteria,attempts,flags)
# OUTPUTS: 
# compactness : It is the sum of squared distance from each point to their corresponding centers.
# labels : This is the label array where each element marked '0', '1', '2', ...
# centers : This is array of centers of clusters.

print("compactness = ",compactness)
print("labels = ",labels)
print("center = ",centers)

# Now convert back into uint8, and make original image
centers = np.uint8(centers)
res = centers[labels.flatten()]
res2 = res.reshape((img.shape))

# Show and save output image
cv2.namedWindow("output", cv2.WINDOW_NORMAL)
cv2.imshow('output',cv2.resize(res2,(960, 960)))
cv2.imwrite("SegmentedImg.jpg", res2)
cv2.waitKey(0)
cv2.destroyAllWindows()
