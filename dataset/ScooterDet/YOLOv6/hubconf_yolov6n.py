def yolov6n(class_names=CLASS_NAMES, device=DEVICE, img_size=640,
    conf_thres=0.25, iou_thres=0.45, max_det=1000):
    return create_model('yolov6n', class_names, device, img_size=img_size,
        conf_thres=conf_thres, iou_thres=iou_thres, max_det=max_det)
