# https://github.com/hpc203/yolox-opencv-dnn/blob/main/main.py
import argparse
import cv2
import numpy as np
import os


class yolox():








if __name__ == '__main__':
    parser = argparse.ArgumentParser("opencv inference sample")
    parser.add_argument("--model", type=str, default="models/yolox_m.onnx", help="Input your onnx model.")
    parser.add_argument("--img", type=str, default='sample.jpg', help="Path to your input image.")
    parser.add_argument("--score_thr", type=float, default=0.3, help="Score threshold to filter the result.")
    parser.add_argument("--classesFile", type=str, default='coco.names', help="Path to your classesFile.")
    parser.add_argument("--with_p6", action="store_true", help="Whether your model uses p6 in FPN/PAN.")
    args = parser.parse_args()
    net = yolox(args.model, args.classesFile, p6=args.with_p6, confThreshold=args.score_thr)
    srcimg = cv2.imread(args.img)
    input = srcimg.copy()


    # Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the
    # timings for each of the layers(in layersTimes)
    cycles = 300
    total_time = 0
    for i in range(cycles):
        srcimg = net.detect(input.copy())
        t, _ = net.net.getPerfProfile()
        total_time += t
        print(f'Cycle [{i + 1}]:\t{t * 1000.0 / cv2.getTickFrequency():.2f}\tms')

    avg_time = total_time / cycles
    window_name = os.path.splitext(os.path.basename(args.model))[0]
    label = 'Average inference time: %.2f ms' % (avg_time * 1000.0 / cv2.getTickFrequency())
    print(f'Model: {window_name}\n{label}')
    cv2.putText(srcimg, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 1, cv2.LINE_AA)

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, srcimg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()