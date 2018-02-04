import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import cv2

train_path = 'data/train'
test_path = 'data/test'


def read_image():
    image_per_class = {}
    for folder_name in os.listdir(train_path):
        folder_path = os.path.join(train_path, folder_name)
        image_per_class[folder_name] = []
        for image_path in glob.glob(os.path.join(folder_path, "*.png")):
            rgb_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            image_per_class[folder_name].append(rgb_image)
    return image_per_class


def binary_image(image):
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
    return mask


def save_image():
    images = read_image()
    for image_class in images:
        num = 0
        if not os.path.exists('binary_images/'+image_class):
            os.makedirs('binary_images/'+image_class)
        for image in images[image_class]:
            image_segment = binary_image(image)
            np.save('binary_images/'+image_class+'/{0}.npy'.format(num), image_segment)
            num += 1


save_image()
