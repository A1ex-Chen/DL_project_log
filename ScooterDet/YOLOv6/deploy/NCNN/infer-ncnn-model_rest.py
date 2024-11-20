import numpy as np
import cv2
import argparse
from numpy import ndarray
from typing import List
import math
import ncnn
import sys
import os

ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
MAJOR, MINOR = map(int, cv2.__version__.split('.')[:2])
assert MAJOR == 4






CLASS_NAMES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
               'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
               'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
               'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
               'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
               'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
               'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
               'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
               'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
               'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
               'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
               'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
               'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush')

CLASS_COLORS = [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230),
                (106, 0, 228), (0, 60, 100), (0, 80, 100), (0, 0, 70),
                (0, 0, 192), (250, 170, 30), (100, 170, 30), (220, 220, 0),
                (175, 116, 175), (250, 0, 30), (165, 42, 42), (255, 77, 255),
                (0, 226, 252), (182, 182, 255), (0, 82, 0), (120, 166, 157),
                (110, 76, 0), (174, 57, 255), (199, 100, 0), (72, 0, 118),
                (255, 179, 240), (0, 125, 92), (209, 0, 151), (188, 208, 182),
                (0, 220, 176), (255, 99, 164), (92, 0, 73), (133, 129, 255),
                (78, 180, 255), (0, 228, 0), (174, 255, 243), (45, 89, 255),
                (134, 134, 103), (145, 148, 174), (255, 208, 186),
                (197, 226, 255), (171, 134, 1), (109, 63, 54), (207, 138, 255),
                (151, 0, 95), (9, 80, 61), (84, 105, 51), (74, 65, 105),
                (166, 196, 102), (208, 195, 210), (255, 109, 65),
                (0, 143, 149), (179, 0, 194), (209, 99, 106), (5, 121, 0),
                (227, 255, 205), (147, 186, 208), (153, 69, 1), (3, 95, 161),
                (163, 255, 0), (119, 0, 170), (0, 182, 199), (0, 165, 120),
                (183, 130, 88), (95, 32, 0), (130, 114, 135), (110, 129, 133),
                (166, 74, 118), (219, 142, 185), (79, 210, 114), (178, 90, 62),
                (65, 70, 15), (127, 167, 115), (59, 105, 106), (142, 108, 45),
                (196, 172, 0), (95, 54, 80), (128, 76, 255), (201, 57, 1),
                (246, 0, 122), (191, 162, 208)]

MASK_COLORS = np.array([(255, 56, 56), (255, 157, 151), (255, 112, 31),
                        (255, 178, 29), (207, 210, 49), (72, 249, 10),
                        (146, 204, 23), (61, 219, 134), (26, 147, 52),
                        (0, 212, 187), (44, 153, 168), (0, 194, 255),
                        (52, 69, 147), (100, 115, 255), (0, 24, 236),
                        (132, 56, 255), (82, 0, 133), (203, 56, 255),
                        (255, 149, 200), (255, 55, 199)],
                       dtype=np.uint8)

CONF_THRES = 0.45
IOU_THRES = 0.65








if __name__ == '__main__':
    main(parse_args())