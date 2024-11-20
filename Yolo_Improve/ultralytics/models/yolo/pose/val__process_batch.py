def _process_batch(self, detections, gt_bboxes, gt_cls, pred_kpts=None,
    gt_kpts=None):
    """
        Return correct prediction matrix.

        Args:
            detections (torch.Tensor): Tensor of shape [N, 6] representing detections.
                Each detection is of the format: x1, y1, x2, y2, conf, class.
            labels (torch.Tensor): Tensor of shape [M, 5] representing labels.
                Each label is of the format: class, x1, y1, x2, y2.
            pred_kpts (torch.Tensor, optional): Tensor of shape [N, 51] representing predicted keypoints.
                51 corresponds to 17 keypoints each with 3 values.
            gt_kpts (torch.Tensor, optional): Tensor of shape [N, 51] representing ground truth keypoints.

        Returns:
            torch.Tensor: Correct prediction matrix of shape [N, 10] for 10 IoU levels.
        """
    if pred_kpts is not None and gt_kpts is not None:
        area = ops.xyxy2xywh(gt_bboxes)[:, 2:].prod(1) * 0.53
        iou = kpt_iou(gt_kpts, pred_kpts, sigma=self.sigma, area=area)
    else:
        iou = box_iou(gt_bboxes, detections[:, :4])
    return self.match_predictions(detections[:, 5], gt_cls, iou)
