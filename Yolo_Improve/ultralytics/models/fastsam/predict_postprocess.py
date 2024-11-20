def postprocess(self, preds, img, orig_imgs):
    """
        Perform post-processing steps on predictions, including non-max suppression and scaling boxes to original image
        size, and returns the final results.

        Args:
            preds (list): The raw output predictions from the model.
            img (torch.Tensor): The processed image tensor.
            orig_imgs (list | torch.Tensor): The original image or list of images.

        Returns:
            (list): A list of Results objects, each containing processed boxes, masks, and other metadata.
        """
    p = ops.non_max_suppression(preds[0], self.args.conf, self.args.iou,
        agnostic=self.args.agnostic_nms, max_det=self.args.max_det, nc=1,
        classes=self.args.classes)
    full_box = torch.zeros(p[0].shape[1], device=p[0].device)
    full_box[2], full_box[3], full_box[4], full_box[6:] = img.shape[3
        ], img.shape[2], 1.0, 1.0
    full_box = full_box.view(1, -1)
    critical_iou_index = bbox_iou(full_box[0][:4], p[0][:, :4], iou_thres=
        0.9, image_shape=img.shape[2:])
    if critical_iou_index.numel() != 0:
        full_box[0][4] = p[0][critical_iou_index][:, 4]
        full_box[0][6:] = p[0][critical_iou_index][:, 6:]
        p[0][critical_iou_index] = full_box
    if not isinstance(orig_imgs, list):
        orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)
    results = []
    proto = preds[1][-1] if len(preds[1]) == 3 else preds[1]
    for i, pred in enumerate(p):
        orig_img = orig_imgs[i]
        img_path = self.batch[0][i]
        if not len(pred):
            masks = None
        elif self.args.retina_masks:
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4],
                orig_img.shape)
            masks = ops.process_mask_native(proto[i], pred[:, 6:], pred[:,
                :4], orig_img.shape[:2])
        else:
            masks = ops.process_mask(proto[i], pred[:, 6:], pred[:, :4],
                img.shape[2:], upsample=True)
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4],
                orig_img.shape)
        results.append(Results(orig_img, path=img_path, names=self.model.
            names, boxes=pred[:, :6], masks=masks))
    return results
