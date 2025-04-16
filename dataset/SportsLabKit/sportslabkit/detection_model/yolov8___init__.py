def __init__(self, model: str='yolov8x', agnostic_nms: bool=False,
    multi_label: bool=False, classes: (list[str] | None)=None, max_det: int
    =1000, amp: bool=False, imgsz: int=640, conf: float=0.25, iou: float=
    0.45, device: str='cpu', verbose: bool=False, augment: bool=False):
    super().__init__(model, agnostic_nms, multi_label, classes, max_det,
        amp, imgsz, conf, iou, device, verbose, augment)
