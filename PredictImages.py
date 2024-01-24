import cv2
import numpy as np
# import matplotlib.pyplot as plt
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
import subprocess

def predict(filepath):
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

    # Resize the image while maintaining the aspect ratio
    img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

    blur = cv2.blur(img,(7,7))

    bblur = cv2.bilateralFilter(blur,9,150,150)

    median_filtered = cv2.medianBlur(bblur, 9) 


    def adptThresholding(image):
        img_gray =cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        adptThresh = cv2.adaptiveThreshold(img_gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
    cv2.THRESH_BINARY,11,2)
        # plt.imshow(adptThresh,cmap='gray')
        return adptThresh

    adpt_thresh_img = adptThresholding(median_filtered)

    def thresholding(image):
        ret, thresh = cv2.threshold(image,127,255,cv2.THRESH_BINARY_INV)
        # plt.imshow(thresh,cmap='gray')
        return thresh

    thresh_img = thresholding(adpt_thresh_img)

    # Apply a median filter
    median_filtered = cv2.medianBlur(thresh_img, 7)  # The second parameter is the kernel size (must be an odd number)

    #Dilation By lines
    kernel =np.ones((5,60), np.uint8)
    dilated = cv2.dilate(median_filtered,kernel,iterations=1)

    (contours, heirarchy) = cv2.findContours(dilated.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    sorted_contours_lines = sorted(contours, key =lambda ctr :cv2.boundingRect(ctr)[1]) #sorts line from top to botton

    img2 = img.copy()

    for ctr in sorted_contours_lines:

        x,y,w,h = cv2.boundingRect(ctr)
        cv2.rectangle(img2,(x,y),(x+w, y+h), (40,100,250),2)

    #Dilation By word
    kernel =np.ones((10,10), np.uint8)
    dilated2 = cv2.dilate(median_filtered,kernel,iterations=1)
    img3 = img.copy()

    # Assuming sorted_contours_lines is a list of contours
    sorted_contours_lines = sorted(sorted_contours_lines, key=lambda line: (cv2.boundingRect(line)[1], cv2.boundingRect(line)[0]))

    # Variable to store predictions with order
    all_predictions_with_order = []

    def process_line(line, line_order):
        x, y, w, h = cv2.boundingRect(line)
        roi_line = dilated2[y:y + h, x:x + w]

        (cnt, _) = cv2.findContours(roi_line.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        sorted_contours_words = sorted(cnt, key=lambda cntr: cv2.boundingRect(cntr)[0])

        with tempfile.TemporaryDirectory() as temp_dir:
            batch_temp_file_paths = []

            for word in sorted_contours_words:
                x2, y2, w2, h2 = cv2.boundingRect(word)
                cv2.rectangle(img3, (x + x2, y + y2), (x + x2 + w2, y + y2 + h2), (0, 50, 100), 2)
                cropped_img = img3[y + y2: y + y2 + h2, x + x2: x + x2 + w2]

                # Save the cropped image to a temporary file
                temp_file_path = os.path.join(temp_dir, "temp_cropped_img{}.png".format(len(batch_temp_file_paths)))
                cv2.imwrite(temp_file_path, cropped_img)
                batch_temp_file_paths.append(temp_file_path)

            # Execute the script with the paths to the temporary files (batch)
            batch_command = ["python", "src/predict.py"] + batch_temp_file_paths
            prediction_batch = subprocess.check_output(batch_command, universal_newlines=True).splitlines()


            # Concatenate predictions with space between words
            line_prediction = " ".join(prediction_batch)
            all_predictions_with_order.append((line_order, line_prediction))

    # Use ThreadPoolExecutor's map method to maintain order
    with ThreadPoolExecutor() as executor:
        for i, line in enumerate(sorted_contours_lines):
            executor.submit(process_line, line, i)


    # Sort predictions based on original order
    all_predictions_with_order.sort(key=lambda x: x[0])

    # Extract predictions for final result
    all_predictions_result = "\n".join(prediction for _, prediction in all_predictions_with_order)

    # Print or use the 'all_predictions_result' variable as needed
    return all_predictions_result