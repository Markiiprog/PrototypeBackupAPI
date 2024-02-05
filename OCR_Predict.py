import cv2
import os
import tempfile
from Preprocessing import ResizeImage, NoiseReduction, adptThresholding, thresholding
from TextDetection import DetectAndSortLines, DetectAndSortWords
from load_OCR import LoadModel
from src.predict import perform_prediction

def process_line(line, line_order, img, dilated2, all_predictions_with_order, crnn, temp_dir):
    try:
        x, y, w, h = cv2.boundingRect(line)
        roi_line = dilated2[y:y + h, x:x + w]

        # Connected Component Analysis (CCA) for word detection
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(roi_line, connectivity=8)

        # Filter out small components (adjust min_size as needed)
        min_size = 500
        valid_stats = [stat for stat in stats[1:] if stat[4] > min_size]

        # Create bounding boxes for valid components
        words = [(stat[0], stat[1], stat[0] + stat[2], stat[1] + stat[3]) for stat in valid_stats]

        # Sort words based on the x-coordinate of the bounding boxes
        sorted_words = sorted(words, key=lambda box: box[0])

        line_prediction = []  # Initialize line prediction as a list

        for word_box in sorted_words:
            x2, y2, x2_end, y2_end = word_box
            cropped_img = img[y + y2: y + y2_end, x + x2: x + x2_end]

            # Save the cropped image to a temporary file
            temp_file_path = os.path.join(temp_dir, "temp_cropped_img{}.png".format(len(all_predictions_with_order)))
            cv2.imwrite(temp_file_path, cropped_img)

            # Execute the script with the path to the temporary file
            prediction = perform_prediction([temp_file_path], crnn, decode_method='beam_search', beam_size=10)

            # Strip leading and trailing whitespaces from individual word predictions
            stripped_prediction = prediction.strip()

            # Append stripped individual word predictions to the line prediction list
            line_prediction.append(stripped_prediction)

        # Concatenate word predictions with space
        line_result = " ".join(line_prediction)

        # Append to the result list only if the line is not empty
        if line_result:
            all_predictions_with_order.append((line_order, line_result))

    except Exception as e:
        print(f"Error processing line {line_order}: {e}")

def Predict(image_filePath):
    img = ResizeImage(image_filePath)
    filtered_img = NoiseReduction(img)
    adpt_thresh_img = adptThresholding(filtered_img)
    thresh_img = thresholding(adpt_thresh_img)
    # Another median Filter to remove noise from thresholding output
    median_filtered = cv2.medianBlur(thresh_img, 5)
    sorted_contours_lines = DetectAndSortLines(median_filtered)
    dilated2 = DetectAndSortWords(median_filtered)
    sorted_contours_lines = sorted(sorted_contours_lines, key=lambda line: (cv2.boundingRect(line)[1], cv2.boundingRect(line)[0]))
    crnn = LoadModel()
    all_predictions_with_order = []

    with tempfile.TemporaryDirectory() as temp_dir:
        for i, line in enumerate(sorted_contours_lines):
            process_line(line, i, img, dilated2, all_predictions_with_order, crnn, temp_dir)

    # Sort predictions based on original order
    all_predictions_with_order.sort(key=lambda x: x[0])

    # Extract predictions for the final result with a single line break after each line
    all_predictions_result = "\n".join(prediction for _, prediction in all_predictions_with_order)

    # Print or use the 'all_predictions_result' variable as needed

    return all_predictions_result
