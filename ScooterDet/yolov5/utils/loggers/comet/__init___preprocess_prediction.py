def preprocess_prediction(self, image, labels, shape, pred):
    nl, _ = labels.shape[0], pred.shape[0]
    if self.opt.single_cls:
        pred[:, 5] = 0
    predn = pred.clone()
    scale_boxes(image.shape[1:], predn[:, :4], shape[0], shape[1])
    labelsn = None
    if nl:
        tbox = xywh2xyxy(labels[:, 1:5])
        scale_boxes(image.shape[1:], tbox, shape[0], shape[1])
        labelsn = torch.cat((labels[:, 0:1], tbox), 1)
        scale_boxes(image.shape[1:], predn[:, :4], shape[0], shape[1])
    return predn, labelsn
