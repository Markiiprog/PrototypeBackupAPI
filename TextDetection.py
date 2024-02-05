import cv2
import numpy as np

#Line Detection
def DetectAndSortLines(median_filtered_image):
    kernel =np.ones((5,60), np.uint8)
    dilated = cv2.dilate(median_filtered_image,kernel,iterations=1)

    (contours, heirarchy) = cv2.findContours(dilated.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    sorted_contours_lines = sorted(contours, key =lambda ctr :cv2.boundingRect(ctr)[1]) #sorts line from top to botton

    return sorted_contours_lines

#Word Detection
def DetectAndSortWords(median_filtered_image):
    height, width = median_filtered_image.shape[:2]

    factor_w = 165
    factor_h = 130
    # Calculate the kernel size based on the factor
    kernel_size = min(height/ factor_h, width/ factor_w)

    kernel_size = int(kernel_size)

    # Create a square kernel with the calculated size
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # Perform dilation
    dilated2 = cv2.dilate(median_filtered_image, kernel, iterations=1)

    return dilated2
