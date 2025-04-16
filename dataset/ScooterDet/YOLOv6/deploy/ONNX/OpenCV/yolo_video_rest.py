# usage: python yolo_video.py --model "./yolov6n.onnx" --source 0

import cv2
import numpy as np
import argparse

INPUT_WIDTH = 640
INPUT_HEIGHT = 640
SCORE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.45
CONFIDENCE_THRESHOLD = 0.2

# Text parameters.
FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.7
THICKNESS = 1

# Colors.
BLACK  = (0,0,0)
BLUE   = (255,178,50)
YELLOW = (0,255,255)







if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='models/yolov6n.onnx', help="Input your onnx model.")
    parser.add_argument('--source', default=0, type=int, help="video source - 0,1,2 ...")
    parser.add_argument('--classesFile', default='coco.names', help="Path to your classesFile.")
    args = parser.parse_args()

    modelWeights, video_source, classesFile = args.model, args.source, args.classesFile
    cap = cv2.VideoCapture(video_source)
    classes = None
    with open(classesFile, 'rt') as f:
        classes = f.read().rstrip('\n').split('\n')

    video()