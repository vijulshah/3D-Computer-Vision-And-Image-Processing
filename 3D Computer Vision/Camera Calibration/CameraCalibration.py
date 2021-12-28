import cv2
import numpy as np
import scipy.io
import glob
import os

OUTPUT_FOLDER = './images/output/'
chessboardSize = (8,6)
frameSize = (3024,4032)

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
objp[:,:2] = np.mgrid[0:chessboardSize[0],0:chessboardSize[1]].T.reshape(-1,2)

size_of_chessboard_squares_mm = 20
objp = objp * size_of_chessboard_squares_mm

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = glob.glob('./images/input/*.jpg')

for i, image in enumerate(images):

    img = cv2.imread(image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, chessboardSize, None)

    print(ret)

    # If found, add object points, image points (after refining them)
    if ret == True:

        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners)

        # Draw and display the corners
        cv2.drawChessboardCorners(gray, chessboardSize, corners2, ret)
        cv2.imwrite(os.path.join(OUTPUT_FOLDER+str(i)+".jpg"), gray)
        cv2.imshow('img', gray)
        cv2.waitKey(1000)

cv2.destroyAllWindows()

############## CALIBRATION #######################################################

ret, cameraMatrix, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, frameSize, None, None)

scipy.io.savemat('camera_calibrated.mat', {
    'camera_matrix_k': cameraMatrix, 
    'rotation_matrix_r': rvecs, 
    'translation_vector_t': tvecs, 
    'distortion_parameter': dist, 
    'points_in_world_3d': objp
})
