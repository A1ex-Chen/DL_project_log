def predict(self, inputs, tf_nms=False, agnostic_nms=False, topk_per_class=
    100, topk_all=100, iou_thres=0.45, conf_thres=0.25):
    y = []
    x = inputs
    for m in self.model.layers:
        if m.f != -1:
            x = y[m.f] if isinstance(m.f, int) else [(x if j == -1 else y[j
                ]) for j in m.f]
        x = m(x)
        y.append(x if m.i in self.savelist else None)
    if tf_nms:
        boxes = self._xywh2xyxy(x[0][..., :4])
        probs = x[0][:, :, 4:5]
        classes = x[0][:, :, 5:]
        scores = probs * classes
        if agnostic_nms:
            nms = AgnosticNMS()((boxes, classes, scores), topk_all,
                iou_thres, conf_thres)
        else:
            boxes = tf.expand_dims(boxes, 2)
            nms = tf.image.combined_non_max_suppression(boxes, scores,
                topk_per_class, topk_all, iou_thres, conf_thres, clip_boxes
                =False)
        return nms,
    return x
