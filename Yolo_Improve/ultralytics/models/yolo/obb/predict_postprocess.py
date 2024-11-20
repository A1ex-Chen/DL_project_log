def postprocess(self, preds, img, orig_imgs):
    """Post-processes predictions and returns a list of Results objects."""
    preds = ops.non_max_suppression(preds, self.args.conf, self.args.iou,
        agnostic=self.args.agnostic_nms, max_det=self.args.max_det, nc=len(
        self.model.names), classes=self.args.classes, rotated=True)
    if not isinstance(orig_imgs, list):
        orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)
    results = []
    for pred, orig_img, img_path in zip(preds, orig_imgs, self.batch[0]):
        rboxes = ops.regularize_rboxes(torch.cat([pred[:, :4], pred[:, -1:]
            ], dim=-1))
        rboxes[:, :4] = ops.scale_boxes(img.shape[2:], rboxes[:, :4],
            orig_img.shape, xywh=True)
        obb = torch.cat([rboxes, pred[:, 4:6]], dim=-1)
        results.append(Results(orig_img, path=img_path, names=self.model.
            names, obb=obb))
    return results
