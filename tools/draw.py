import cv2
import os
import numpy as np
import glob


results = "/data/vpta/dataset/results/VOC2012/Main/car.txt"
source_path = "/data/vpta/dataset/VOC2012/JPEGImages-test"
dest_path = "/data/vpta/dataset/results/VOC2012/output"


def draw():
    with open(results, 'r') as f:
        lines = f.readlines()

    for line in lines:
        split_line = line.split()
        name = split_line[0]
        conf = float(split_line[1])
        if conf < 0.5:
            continue
        poly = np.array(split_line[2:], float).reshape((-1,1, 2)).astype('int32')

        image_name = os.path.join(source_path, f"{name}.jpg")
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


if __name__ == "__main__":
    for file in glob.glob("/data/vpta/dataset/VOC2012/labelTxt/*.txt"):
        file_name = file.split('/')[-1]
        file_num = file_name.split('.')[0]
        with open(file, 'r') as f:
            lines = f.readlines()
        poly = np.array(lines[2].split()[:-2], float).reshape((-1, 1, 2)).astype('int32')

        image_name = os.path.join(source_path, f"{file_num}.jpg")
        dst_name = os.path.join(dest_path, f"{file_num}.jpg")
        image = cv2.imread(image_name)

        print(image_name, poly)

        # color, thickness and isClosed
        color = (255, 0, 0)
        thickness = 2
        isClosed = True

        # drawPolyline
        image = cv2.polylines(image, [poly], isClosed, color, thickness,)
        cv2.imwrite(dst_name, image)
