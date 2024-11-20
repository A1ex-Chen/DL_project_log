def create_model(model_name, class_names=CLASS_NAMES, device=DEVICE,
    img_size=640, conf_thres=0.25, iou_thres=0.45, max_det=1000):
    if not os.path.exists(str(PATH_YOLOv6 / 'weights')):
        os.mkdir(str(PATH_YOLOv6 / 'weights'))
    if not os.path.exists(str(PATH_YOLOv6 / 'weights') + f'/{model_name}.pt'):
        torch.hub.load_state_dict_from_url(
            f'https://github.com/meituan/YOLOv6/releases/download/0.3.0/{model_name}.pt'
            , str(PATH_YOLOv6 / 'weights'))
    return Detector(str(PATH_YOLOv6 / 'weights') + f'/{model_name}.pt',
        class_names, device, img_size=img_size, conf_thres=conf_thres,
        iou_thres=iou_thres, max_det=max_det)
