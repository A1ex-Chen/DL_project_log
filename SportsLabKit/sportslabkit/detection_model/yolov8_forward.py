def forward(self, x, **kwargs):

    def to_dict(res):
        if len(res) == 0:
            return [{}]
        return [{'bbox_left': r[0] - r[2] / 2, 'bbox_top': r[1] - r[3] / 2,
            'bbox_width': r[2], 'bbox_height': r[3], 'conf': r[4], 'class':
            r[5]} for r in res]
    x = [_x[..., ::-1] for _x in x]
    results = self.model(x, agnostic_nms=kwargs.get('agnostic_nms', self.
        agnostic_nms), classes=kwargs.get('classes', self.classes), max_det
        =kwargs.get('max_det', self.max_det), imgsz=kwargs.get('imgsz',
        self.imgsz), conf=kwargs.get('conf', self.conf), iou=kwargs.get(
        'iou', self.iou), device=kwargs.get('device', self.device), verbose
        =kwargs.get('verbose', self.verbose), task='detect', augment=kwargs
        .get('augment', self.augment))
    preds = []
    for result in results:
        xywh = result.boxes.xywh.detach().cpu().numpy()
        conf = result.boxes.conf.detach().cpu().numpy()
        cls = result.boxes.cls.detach().cpu().numpy()
        res = np.concatenate([xywh, conf.reshape(-1, 1), cls.reshape(-1, 1)
            ], axis=1)
        preds.append(to_dict(res))
    return preds
