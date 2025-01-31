import torch
from utils import TryExcept
from utils.augmentations import letterbox
from utils.general import (LOGGER, ROOT, Profile, non_max_suppression, scale_boxes, xywh2xyxy,
                           xyxy2xywh, embedding_distance)
from utils.torch_utils import select_device
import numpy as np
from config import config
import cv2
import os
from norfair import Tracker, Video, draw_points, draw_tracked_objects, Detection
from norfair.filter import OptimizedKalmanFilterFactory
from models import DetectBackend, BodyFeatureExtractBackend
import time
from sender import send_frame, send_feature
import pika
import _thread
import serial
import json
from threading import Thread

class VideoStream:
    def __init__(self, src=0):
        # initialize the video camera stream and read the first frame
        # from the stream
        self.stream = cv2.VideoCapture(src)
        (self.grabbed, self.frame) = self.stream.read()
        # initialize the variable used to indicate if the thread should
        # be stopped
        self.stopped = False
    def start(self):
        # start the thread to read frames from the video stream
        Thread(target=self.update, args=()).start()
        return self
    def update(self):
        # keep looping infinitely until the thread is stopped
        while True:
            # if the thread indicator variable is set, stop the thread
            if self.stopped:
                return
            # otherwise, read the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()
    def read(self):
        # return the frame most recently read
        return self.grabbed, self.frame
    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True
    def isOpened(self):
        return self.stream.isOpened()


def ReadData(nameThread):
    global humidity
    global temperature
    global ppm
    print("Create thread read data")
    humidity = 0.0
    temperature = 0.0
    ppm = 0.0
    ser = serial.Serial(port= '/dev/ttyACM0', baudrate=115200)
    
    time.sleep(8)

    while True:
        try:   
            time.sleep(2) 
            s = ser.readline()
            data = s.decode("utf-8")
            j = json.loads(data)
            humidity = j["humidity"]
            temperature = j["temperature"]
            ppm = j["ppm"]
      
        except:
            print("error")

# Read video input
cap = VideoStream(src=config.source).start()
print('Camera Ready?', cap.isOpened())
if cap.isOpened() == False:
    os._exit(1)


_thread.start_new_thread(ReadData, ("Read Data",))

# Load detection model
device = select_device(config.device)
model = DetectBackend(config.weights, device=device, data=config.classes, fp16=config.fp16)
stride, names = model.stride, model.names
model.warmup(imgsz=(1, 3, config.height, config.width))  # warmup

# Load extracting model
body_model = BodyFeatureExtractBackend(config.body_extract_model)
body_model.warmup()

count, dt = -1, (Profile(), Profile(), Profile(), Profile())
frame_time, ft_time = 0, 0
LOGGER.info('Creating Tracker...')
tracker = Tracker(
        initialization_delay=config.initialization_delay,
        distance_function="custom",
        hit_counter_max=config.hit_counter_max,
        filter_factory=OptimizedKalmanFilterFactory(),
        distance_threshold=config.distance_threshold,
        past_detections_length=config.past_detections_length,
        reid_distance_function=embedding_distance,
        reid_distance_threshold=config.reid_distance_threshold,
        reid_hit_counter_max=config.reid_hit_counter_max,
        )


global humidity
global temperature
global ppm
LOGGER.info('Start running...')
while cap.isOpened():
    try:
        start_time = time.time()
        ret, ori_im = cap.read()
        if ret == False:
            break
        count += 1
        if count % config.skip_period == 0:
            # Detection preprocess
            with dt[0]:
                im = letterbox(ori_im, (config.height, config.width), stride=stride, auto=False)[0]
                im = im[np.newaxis, ...]
                im = im[..., ::-1].transpose((0, 3, 1, 2))
                im = np.ascontiguousarray(im)
                im = torch.from_numpy(im).to(model.device)
                im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
                im /= 255  # 0 - 255 to 0.0 - 1.0
                if len(im.shape) == 3:
                    im = im[None]  # expand for batch dim

            # Detection Inference
            with dt[1]:
                pred = model(im)
            
            # NMS
            with dt[2]:
                det = non_max_suppression(pred, config.conf_thres, config.iou_thres, 0, False, max_det=config.max_det)[0]

            s = ''
            s += '%gx%g ' % im.shape[2:]  # print string
            if len(det):
                dect_ls = []
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], ori_im.shape).round()
                with dt[3]:
                    for *xyxy, conf, cls in det:
                        xmin, ymin, xmax, ymax = xyxy
                        xmin, ymin, xmax, ymax = round(xmin.item()), round(ymin.item()), round(xmax.item()), round(ymax.item())
                        if (ymax-ymin)/(xmax-xmin) > 10 or (ymax-ymin)/(xmax-xmin) < 0.9:
                            det_pred = Detection(
                                    points=np.vstack(
                                        (
                                            [xmin, ymin],
                                            [xmax, ymax],
                                            )
                                        ),
                                    data=[xmin/ori_im.shape[1], ymin/ori_im.shape[0], xmax/ori_im.shape[1], ymax/ori_im.shape[0]],
                                    label=names[int(cls)],
                                    embedding=None,
                                    )
                        else:
                            det_pred = Detection(
                                    points=np.vstack(
                                        (
                                            [xmin, ymin],
                                            [xmax, ymax],
                                            )
                                        ),
                                    data=[xmin/ori_im.shape[1], ymin/ori_im.shape[0], xmax/ori_im.shape[1], ymax/ori_im.shape[0]],
                                    label=names[int(cls)],
                                    embedding=body_model.extract(ori_im[ymin:ymax, xmin:xmax]),
                                    )
                        dect_ls.append(det_pred)
                    tracked_objects = tracker.update(detections=dect_ls, period=config.skip_period)
                # Print results
          #      for c in det[:, 5].unique():
           #         n = (det[:, 5] == c).sum()  # detections per class
            #        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
            else:
                with dt[3]:
                    tracked_objects = tracker.update(period=config.skip_period)
        else:
            with dt[3]:
                tracked_objects = tracker.update()
        
        frame_time = frame_time + time.time() - start_time
        ft_time = ft_time + time.time() - start_time
        if frame_time > config.frame_interval:
            send_frame(ori_im, humidity, temperature, ppm, len(tracked_objects))
            frame_time = 0
        if ft_time > config.feature_interval:
            send_feature(tracked_objects)
            ft_time = 0
        LOGGER.info(f"FPS: {1/(time.time()-start_time)} fps")
        # Print time (inference-only)
        # LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[0].dt * 1E3:.1f}ms, {dt[1].dt * 1E3:.1f}ms, {dt[2].dt * 1E3:.1f}ms, {dt[3].dt * 1E3:.1f}ms, {1/(dt[0].dt+dt[1].dt+dt[2].dt+dt[3].dt):.1f}fps")
    except KeyboardInterrupt:
        break
cap.stop()
cap.stream.release()
del cap
