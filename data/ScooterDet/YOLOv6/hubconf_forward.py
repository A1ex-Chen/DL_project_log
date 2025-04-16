def forward(self, x, src_shape):
    pred_results = super().forward(x)
    classes = None
    det = non_max_suppression(pred_results, self.conf_thres, self.iou_thres,
        classes, agnostic=False, max_det=self.max_det)[0]
    det[:, :4] = Inferer.rescale(x.shape[2:], det[:, :4], src_shape).round()
    boxes = det[:, :4]
    scores = det[:, 4]
    labels = det[:, 5].long()
    prediction = {'boxes': boxes, 'scores': scores, 'labels': labels}
    return prediction
