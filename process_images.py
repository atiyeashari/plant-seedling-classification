import numpy as np
import os
import glob

import cv2

train_path = 'data/train'
test_path = 'data/test'

image_per_class = {}
for folder_name in os.listdir(train_path):
    folder_path = os.path.join(train_path, folder_name)
    image_per_class[folder_name] = []
    for image_path in glob.glob(os.path.join(folder_path, "*.png")):
        rgb_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image_per_class[folder_name].append(rgb_image)


def create_image_segment(image):
    # Convert BGR to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # define range of green color in HSV
    lower_green = np.array([25, 100, 50])
    upper_green = np.array([95, 255, 255])
    # Threshold the HSV image to get only green colors
    mask = cv2.inRange(hsv, lower_green, upper_green)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    # Remove the noise on plant's image
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(image, image, mask=mask)
    return res


def sharpen_image(image):
    image_blurred = cv2.GaussianBlur(image, (0, 0), 3)
    image_sharp = cv2.addWeighted(image, 1.5, image_blurred, -0.5, 0)
    return image_sharp
