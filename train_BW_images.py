import numpy as np
import glob
import os
import cv2
from numpy.core.umath import minimum

train_path = 'binary_images'


def load_images():
    image_per_class = {}
    for folder_name in os.listdir(train_path):
        folder_path = os.path.join(train_path, folder_name)
        image_per_class[folder_name] = []
        for image_path in glob.glob(os.path.join(folder_path, "*.npy")):
            image = np.load(image_path)
            image = cv2.resize(image, (100, 100), 0, 0, interpolation=cv2.INTER_AREA)
            image_per_class[folder_name].append(image)
    return image_per_class


images = load_images()