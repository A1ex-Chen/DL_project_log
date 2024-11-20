def custom(ckpt_path, class_names, device=DEVICE, img_size=640, conf_thres=
    0.25, iou_thres=0.45, max_det=1000):
    return Detector(ckpt_path, class_names, device, img_size=img_size,
        conf_thres=conf_thres, iou_thres=iou_thres, max_det=max_det)
