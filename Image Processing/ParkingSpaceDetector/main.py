import cv2
import pickle
import numpy as np
import cvzone

cap = cv2.VideoCapture('carPark.mp4')

with open('ObjectPos', 'rb') as f:
    posList = pickle.load(f)
    
width, height = 107, 48

def checkParkingSpace(imgPro):
    spaceCounter = 0

    # detect car
    for pos in posList:
        x, y = pos
        imgCrop = imgPro[y:y+height, x:x+width]
        # cv2.imshow(str(x * y), imgCrop)
        count = cv2.countNonZero(imgCrop)

        if count < 900:
            rectColor = (0, 255, 0)
            thickness = 5
            spaceCounter += 1
        else:
            rectColor = (0, 0, 255)
            thickness = 2
        
        cv2.rectangle(img, pos, (pos[0]+width, pos[1]+height), rectColor, thickness)
        cvzone.putTextRect(img, str(count), (x, y+height - 3), scale=1, thickness=2, offset=0, colorR=rectColor)
    
    cvzone.putTextRect(img, f'Free: {spaceCounter}/{len(posList)}', (100, 50), scale=2, thickness=5, offset=20, colorR=(0,255,0))

while True:
    if(cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    success, img = cap.read()
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (3,3), 1)
    # Gaussian blurring is commonly used when reducing the size of an image. When downsampling an image, it is common to apply a low-pass filter to the image prior to resampling. This is to ensure that spurious high-frequency information does not appear in the downsampled image
    imgThreshold = cv2.adaptiveThreshold(imgBlur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 16)
    imgMedian = cv2.medianBlur(imgThreshold, 5)
    # image removes “outlier” pixels that may be noise in the image.
    kernel = np.ones((3, 3), np.uint8)
    imgDialate = cv2.dilate(imgMedian, kernel, iterations=1)
    # it increases the white region in the image or size of foreground object increases. Normally, in cases like noise removal it is followed by dilation

    checkParkingSpace(imgDialate)
    cv2.imshow("Image", img)
    cv2.waitKey(10)
