import cv2

def read_and_preprocess_image(file_path, target_dpi=600):
    img = cv2.imread(file_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    h, w, c = img.shape
    image_width_inches = w / target_dpi
    image_height_inches = h / target_dpi
    new_width = int(image_width_inches * target_dpi)
    new_height = int(image_height_inches * target_dpi)

    img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

    blur = cv2.blur(img, (7, 7))
    bblur = cv2.bilateralFilter(blur, 9, 150, 150)
    median_filtered = cv2.medianBlur(bblur, 9) 

    return median_filtered

def adpt_thresholding(image):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    adpt_thresh = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    return adpt_thresh

def thresholding(image):
    ret, thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
    return thresh