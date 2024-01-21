import cv2
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
import subprocess


def process_line(line, img, dilated2):
    x, y, w, h = cv2.boundingRect(line)
    roi_line = dilated2[y:y + h, x:x + w]

    (cnt, _) = cv2.findContours(roi_line.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    sorted_contours_words = sorted(cnt, key=lambda cntr: cv2.boundingRect(cntr)[0])

    with tempfile.TemporaryDirectory() as temp_dir:
        batch_temp_file_paths = []

        for word in sorted_contours_words:
            x2, y2, w2, h2 = cv2.boundingRect(word)
            cv2.rectangle(img, (x + x2, y + y2), (x + x2 + w2, y + y2 + h2), (0, 50, 100), 2)
            cropped_img = img[y + y2: y + y2 + h2, x + x2: x + x2 + w2]
            
            temp_file_path = os.path.join(temp_dir, "temp_cropped_img{}.png".format(len(batch_temp_file_paths)))
            cv2.imwrite(temp_file_path, cropped_img)
            batch_temp_file_paths.append(temp_file_path)

        batch_command = ["python", "src/predict.py"] + batch_temp_file_paths
        prediction_batch = subprocess.check_output(batch_command, universal_newlines=True).splitlines()

        line_prediction = " ".join(prediction_batch)

        return line_prediction
    
def perform_ocr(sorted_contours_lines, img_copy, dilated):
    with ThreadPoolExecutor() as executor:
        all_predictions_with_order = list(executor.map(process_line, sorted_contours_lines, [img_copy] * len(sorted_contours_lines), [dilated] * len(sorted_contours_lines)))

    return "\n".join(all_predictions_with_order)