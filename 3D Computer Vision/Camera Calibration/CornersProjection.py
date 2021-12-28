import cv2
import numpy as np
import scipy.io as io
import glob

base_folder = './images/input/'
data = io.loadmat('camera_calibrated.mat')

imgs_list = []
images = glob.glob('./images/input/*.jpg')

for image in images:
    img = cv2.imread(image)
    imgs_list.append(img)

imgs = np.asarray(imgs_list)