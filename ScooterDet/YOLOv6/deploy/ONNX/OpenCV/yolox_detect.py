def detect(self, srcimg):
    img, ratio = self.preprocess(srcimg)
    blob = cv2.dnn.blobFromImage(img)
    self.net.setInput(blob)
    outs = self.net.forward(self.net.getUnconnectedOutLayersNames())
    predictions = self.demo_postprocess(outs[0])[0]
    boxes = predictions[:, :4]
    scores = predictions[:, 4:5] * predictions[:, 5:]
    boxes_xyxy = np.ones_like(boxes)
    boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.0
    boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.0
    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.0
    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.0
    boxes_xyxy /= ratio
    dets = self.multiclass_nms(boxes_xyxy, scores)
    if dets is not None:
        final_boxes, final_scores, final_cls_inds = dets[:, :4], dets[:, 4
            ], dets[:, 5]
        srcimg = self.vis(srcimg, final_boxes, final_scores, final_cls_inds)
    return srcimg
