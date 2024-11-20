def postprocess(self, preds, img, orig_imgs):
    """
        Post-processes SAM's inference outputs to generate object detection masks and bounding boxes.

        The method scales masks and boxes to the original image size and applies a threshold to the mask predictions.
        The SAM model uses advanced architecture and promptable segmentation tasks to achieve real-time performance.

        Args:
            preds (tuple): The output from SAM model inference, containing masks, scores, and optional bounding boxes.
            img (torch.Tensor): The processed input image tensor.
            orig_imgs (list | torch.Tensor): The original, unprocessed images.

        Returns:
            (list): List of Results objects containing detection masks, bounding boxes, and other metadata.
        """
    pred_masks, pred_scores = preds[:2]
    pred_bboxes = preds[2] if self.segment_all else None
    names = dict(enumerate(str(i) for i in range(len(pred_masks))))
    if not isinstance(orig_imgs, list):
        orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)
    results = []
    for i, masks in enumerate([pred_masks]):
        orig_img = orig_imgs[i]
        if pred_bboxes is not None:
            pred_bboxes = ops.scale_boxes(img.shape[2:], pred_bboxes.float(
                ), orig_img.shape, padding=False)
            cls = torch.arange(len(pred_masks), dtype=torch.int32, device=
                pred_masks.device)
            pred_bboxes = torch.cat([pred_bboxes, pred_scores[:, None], cls
                [:, None]], dim=-1)
        masks = ops.scale_masks(masks[None].float(), orig_img.shape[:2],
            padding=False)[0]
        masks = masks > self.model.mask_threshold
        img_path = self.batch[0][i]
        results.append(Results(orig_img, path=img_path, names=names, masks=
            masks, boxes=pred_bboxes))
    self.segment_all = False
    return results
