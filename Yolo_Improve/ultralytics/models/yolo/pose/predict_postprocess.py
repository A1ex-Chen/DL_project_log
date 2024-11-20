def postprocess(self, preds, img, orig_imgs):
    """Return detection results for a given input image or list of images."""
    preds = ops.non_max_suppression(preds, self.args.conf, self.args.iou,
        agnostic=self.args.agnostic_nms, max_det=self.args.max_det, classes
        =self.args.classes, nc=len(self.model.names))
    if not isinstance(orig_imgs, list):
        orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)
    results = []
    for i, pred in enumerate(preds):
        orig_img = orig_imgs[i]
        pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.
            shape).round()
        pred_kpts = pred[:, 6:].view(len(pred), *self.model.kpt_shape) if len(
            pred) else pred[:, 6:]
        pred_kpts = ops.scale_coords(img.shape[2:], pred_kpts, orig_img.shape)
        img_path = self.batch[0][i]
        results.append(Results(orig_img, path=img_path, names=self.model.
            names, boxes=pred[:, :6], keypoints=pred_kpts))
    return results
