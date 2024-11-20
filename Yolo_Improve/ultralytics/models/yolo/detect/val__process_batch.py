def _process_batch(self, detections, gt_bboxes, gt_cls):
    """
        Return correct prediction matrix.

        Args:
            detections (torch.Tensor): Tensor of shape [N, 6] representing detections.
                Each detection is of the format: x1, y1, x2, y2, conf, class.
            labels (torch.Tensor): Tensor of shape [M, 5] representing labels.
                Each label is of the format: class, x1, y1, x2, y2.

        Returns:
            (torch.Tensor): Correct prediction matrix of shape [N, 10] for 10 IoU levels.
        """
    iou = box_iou(gt_bboxes, detections[:, :4])
    return self.match_predictions(detections[:, 5], gt_cls, iou)
