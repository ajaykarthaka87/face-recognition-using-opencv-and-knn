# 2_prepare_data.py
import os
import cv2
import numpy as np

data = []
labels = []
label_map = {}
label_id = 0

for person in os.listdir("dataset"):
    person_path = os.path.join("dataset", person)
    if not os.path.isdir(person_path):
        continue

    if person not in label_map:
        label_map[person] = label_id
        label_id += 1

    for image_name in os.listdir(person_path):
        img_path = os.path.join(person_path, image_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (100, 100))
        data.append(img.flatten())
        labels.append(label_map[person])

np.save("data.npy", np.array(data))
np.save("labels.npy", np.array(labels))
np.save("label_map.npy", label_map)
