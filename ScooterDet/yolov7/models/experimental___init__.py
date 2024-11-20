def __init__(self, model, max_obj=100, iou_thres=0.45, score_thres=0.25,
    max_wh=None, device=None, n_classes=80):
    super().__init__()
    device = device if device else torch.device('cpu')
    assert isinstance(max_wh, int) or max_wh is None
    self.model = model.to(device)
    self.model.model[-1].end2end = True
    self.patch_model = ONNX_TRT if max_wh is None else ONNX_ORT
    self.end2end = self.patch_model(max_obj, iou_thres, score_thres, max_wh,
        device, n_classes)
    self.end2end.eval()
