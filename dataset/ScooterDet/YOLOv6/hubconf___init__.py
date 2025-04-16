def __init__(self, ckpt_path, class_names, device, img_size=640, conf_thres
    =0.25, iou_thres=0.45, max_det=1000):
    super().__init__(ckpt_path, device)
    self.class_names = class_names
    self.model.float()
    self.device = device
    self.img_size = check_img_size(img_size)
    self.conf_thres = conf_thres
    self.iou_thres = iou_thres
    self.max_det = max_det
