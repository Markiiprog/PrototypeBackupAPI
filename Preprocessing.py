import cv2
import numpy as np
import matplotlib.pyplot as plt

def ResizeImage(filepath):
    img = cv2.imread(filepath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


    norm_img = np.zeros((img.shape[0], img.shape[1]))
    img = cv2.normalize(img, norm_img, 0, 255, cv2.NORM_MINMAX)

    # Define the desired DPI
    target_dpi = 600

    # Get the current dimensions of the image
    h, w, c = img.shape

    # Calculate the physical size of the image in inches
    image_width_inches = w / target_dpi
    image_height_inches = h / target_dpi

    # Calculate the new dimensions to achieve the desired DPI
    new_width = int(image_width_inches * target_dpi)
    new_height = int(image_height_inches * target_dpi)

    img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

    # Resize the image while maintaining the aspect ratio
    return img


def NoiseReduction(image):

    blurred_img = cv2.GaussianBlur(image, (5, 5), 0)
    blur = cv2.bilateralFilter(blurred_img,7,100,100)
    median_filtered = cv2.medianBlur(blur, 5) 

    return median_filtered


def adptThresholding(image):
    img_gray =cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    adptThresh = cv2.adaptiveThreshold(img_gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
cv2.THRESH_BINARY,11,2)
    plt.imshow(adptThresh,cmap='gray')

    return adptThresh

def thresholding(image):
    ret, thresh = cv2.threshold(image,127,255,cv2.THRESH_BINARY_INV)
    plt.imshow(thresh,cmap='gray')
    return thresh



