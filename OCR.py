import easyocr
import cv2

def perform_ocr(image_path,reader):
    
    recognized_text =""

    # Read the image using OpenCV
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    detection_result = reader.detect(image,
                                    slope_ths=1,
                                    height_ths =1,
                                    width_ths =1
                                 )
    text_coordinates = detection_result[0][0]


    results = reader.recognize(image,
                               horizontal_list=text_coordinates,
                               free_list=[],
                               detail=0,
                               batch_size = 16,
                               paragraph = True
                               )

    final_results = " ".join(results)
    return final_results