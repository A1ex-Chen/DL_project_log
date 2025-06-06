def postprocess(self, preds_in, img, orig_imgs):
    """Postprocess predictions and returns a list of Results objects."""
    boxes = ops.xyxy2xywh(preds_in[0][0])
    preds = torch.cat((boxes, preds_in[0][1]), -1).permute(0, 2, 1)
    preds = ops.non_max_suppression(preds, self.args.conf, self.args.iou,
        agnostic=self.args.agnostic_nms, max_det=self.args.max_det, classes
        =self.args.classes)
    if not isinstance(orig_imgs, list):
        orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)
    results = []
    for i, pred in enumerate(preds):
        orig_img = orig_imgs[i]
        pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.
            shape)
        img_path = self.batch[0][i]
        results.append(Results(orig_img, path=img_path, names=self.model.
            names, boxes=pred))
    return results
