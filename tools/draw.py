import cv2
import os
import numpy as np


results = "/data/vpta/voc_dataset/results/VOC2012/Main/car.txt"
source_path = "/data/vpta/voc_dataset/VOC2012/JPEGImages-test"
dest_path = "/data/vpta/voc_dataset/VOC2012/output"

with open(results, 'r') as f:
    lines = f.readlines()

for line in lines:
    split_line = line.split()
    name = split_line[0]
    conf = split_line[1]
    poly = np.array(split_line[2:], float).reshape((-1,1, 2)).astype('int32')

    image_name = os.path.join(source_path, f"{name}.png")
    dst_name = os.path.join(dest_path, f"{name}.jpg")
    image = cv2.imread(image_name)

    print(image_name, conf, poly)

    # color, thickness and isClosed
    color = (255, 0, 0)
    thickness = 2
    isClosed = True

    # drawPolyline
    image = cv2.polylines(image, [poly], isClosed, color, thickness,)
    cv2.imwrite(dst_name, image)
    # break
