def _process_batch(self, detections, gt_bboxes, gt_cls, pred_masks=None,
    gt_masks=None, overlap=False, masks=False):
    """
        Return correct prediction matrix.

        Args:
            detections (array[N, 6]), x1, y1, x2, y2, conf, class
            labels (array[M, 5]), class, x1, y1, x2, y2

        Returns:
            correct (array[N, 10]), for 10 IoU levels
        """
    if masks:
        if overlap:
            nl = len(gt_cls)
            index = torch.arange(nl, device=gt_masks.device).view(nl, 1, 1) + 1
            gt_masks = gt_masks.repeat(nl, 1, 1)
            gt_masks = torch.where(gt_masks == index, 1.0, 0.0)
        if gt_masks.shape[1:] != pred_masks.shape[1:]:
            gt_masks = F.interpolate(gt_masks[None], pred_masks.shape[1:],
                mode='bilinear', align_corners=False)[0]
            gt_masks = gt_masks.gt_(0.5)
        iou = mask_iou(gt_masks.view(gt_masks.shape[0], -1), pred_masks.
            view(pred_masks.shape[0], -1))
    else:
        iou = box_iou(gt_bboxes, detections[:, :4])
    return self.match_predictions(detections[:, 5], gt_cls, iou)
