def __init__(self, model, max_obj=100, iou_thres=0.45, score_thres=0.25,
    device=None, ort=False, trt_version=8, with_preprocess=False):
    super().__init__()
    device = device if device else torch.device('cpu')
    self.with_preprocess = with_preprocess
    self.model = model.to(device)
    TRT = ONNX_TRT8 if trt_version >= 8 else ONNX_TRT7
    self.patch_model = ONNX_ORT if ort else TRT
    self.end2end = self.patch_model(max_obj, iou_thres, score_thres, device)
    self.end2end.eval()
