# coding=utf-8
# Description:  visualize yolo label image.

import argparse
import os
import cv2
import numpy as np

IMG_FORMATS = ["bmp", "jpg", "jpeg", "png", "tif", "tiff", "dng", "webp", "mpo"]
IMG_FORMATS.extend([f.upper() for f in IMG_FORMATS])




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', default='VOCdevkit/voc_07_12/images')
    parser.add_argument('--label_dir', default='VOCdevkit/voc_07_12/labels')
    parser.add_argument('--class_names', default=['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
                        'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'])

    args = parser.parse_args()
    print(args)

    main(args)