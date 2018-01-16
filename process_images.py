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
