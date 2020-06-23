import cv2
import numpy as np
import os
import sys
import tensorflow as tf

from sklearn.model_selection import train_test_split

images = []
labels = []

data_dir = 'gtsrb-small'

 # Iterate through sign folders in directory:
for foldername in os.listdir(data_dir):
    try:
      int(foldername)
    except ValueError:
      print("Warning! Non-integer folder name in data directory! Skipping...")
      continue
    # Iterate through images in each folder
    for filename in os.listdir(os.path.join(data_dir, foldername)):
        # Open each image, save np array and label to lists
        img = cv2.imread(os.path.join(data_dir, foldername, filename))
        images.append(img)
        labels.append(int(foldername))

print(len(images), len(labels))

