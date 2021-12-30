import cv2
import numpy as np
import os

def convolution(img, kernel):
    convolution_matrix = np.zeros(img.shape)
    for i in range(img.shape[0]-1):
        for j in range(img.shape[1]-1):
            sum = 0
            for m in range(kernel.shape[0]):
                for n in range(kernel.shape[1]):
                    sum += kernel[m][n] * img[i+m-1][j+n-1]
            convolution_matrix[i-1][j-1] = sum
    return convolution_matrix

def pad_image(image):
    imagePadded = np.asarray([[ 0 for x in range(0,image.shape[1] + 2)] for y in range(0,image.shape[0] + 2)], dtype =np.uint8)
    imagePadded[1:(imagePadded.shape[0]-1), 1:(imagePadded.shape[1]-1)] = image 
    return imagePadded

def sobel_edge_detection(imgx, imgy):
    img_copy = np.zeros(imgx.shape)    
    list = []
    for i in range(imgx.shape[0]):
        for j in range(imgx.shape[1]):
            q = np.sqrt(imgx[i][j]**2 + imgy[i][j]**2)
            img_copy[i][j] = q
            list.append(q)
    img_copy /= max(list)
    return img_copy

inputImage = cv2.imread("./images/sobel/input/sobel_input.jpg",0)
OUTPUT_FOLDER = "./images/sobel/output/"

scale_percent = 20 # percent of original size
width = int(inputImage.shape[1] * scale_percent / 100)
height = int(inputImage.shape[0] * scale_percent / 100)
dim = (width, height)

inputImage = cv2.resize(inputImage, dim, interpolation = cv2.INTER_AREA)

kernel_in_x_direction = np.array([[-1,0,1], [-2,0,2], [-1,0,1]])
kernel_in_y_direction = kernel_in_x_direction.T

# padding image with zeros along last pixel rows and columns of an image so that kernel can be applied to corners of the image easily
inputImage = pad_image(inputImage)

gradient_in_x_direction = convolution(inputImage, kernel_in_x_direction)
cv2.imshow("gradient_in_x_direction",gradient_in_x_direction)

gradient_in_y_direction = convolution(inputImage, kernel_in_y_direction)
cv2.imshow("gradient_in_y_direction",gradient_in_y_direction)

sobel_edge = sobel_edge_detection(gradient_in_x_direction, gradient_in_y_direction)
cv2.imwrite(os.path.join(OUTPUT_FOLDER+"sobel_output.jpg"), sobel_edge)
cv2.imshow("sobel_edge", sobel_edge)

cv2.waitKey(0)
cv2.destroyAllWindows()
