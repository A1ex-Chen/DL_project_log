# YOLOv3 🚀 by Ultralytics, AGPL-3.0 license
"""
PyTorch Hub models https://pytorch.org/hub/ultralytics_yolov5

Usage:
    import torch
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # official model
    model = torch.hub.load('ultralytics/yolov5:master', 'yolov5s')  # from branch
    model = torch.hub.load('ultralytics/yolov5', 'custom', 'yolov5s.pt')  # custom/local model
    model = torch.hub.load('.', 'custom', 'yolov5s.pt', source='local')  # local repo
"""

import torch


























if __name__ == "__main__":
    import argparse
    from pathlib import Path

    import numpy as np
    from PIL import Image

    from utils.general import cv2, print_args

    # Argparser
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="yolov5s", help="model name")
    opt = parser.parse_args()
    print_args(vars(opt))

    # Model
    model = _create(name=opt.model, pretrained=True, channels=3, classes=80, autoshape=True, verbose=True)
    # model = custom(path='path/to/model.pt')  # custom

    # Images
    imgs = [
        "data/images/zidane.jpg",  # filename
        Path("data/images/zidane.jpg"),  # Path
        "https://ultralytics.com/images/zidane.jpg",  # URI
        cv2.imread("data/images/bus.jpg")[:, :, ::-1],  # OpenCV
        Image.open("data/images/bus.jpg"),  # PIL
        np.zeros((320, 640, 3)),
    ]  # numpy

    # Inference
    results = model(imgs, size=320)  # batched inference

    # Results
    results.print()
    results.save()