def _process_batch(self, detections, gt_bboxes, gt_cls):
    """
        Return correct prediction matrix.

        Args:
            detections (torch.Tensor): Tensor of shape [N, 7] representing detections.
                Each detection is of the format: x1, y1, x2, y2, conf, class, angle.
            gt_bboxes (torch.Tensor): Tensor of shape [M, 5] representing rotated boxes.
                Each box is of the format: x1, y1, x2, y2, angle.
            labels (torch.Tensor): Tensor of shape [M] representing labels.

        Returns:
            (torch.Tensor): Correct prediction matrix of shape [N, 10] for 10 IoU levels.
        """
    iou = batch_probiou(gt_bboxes, torch.cat([detections[:, :4], detections
        [:, -1:]], dim=-1))
    return self.match_predictions(detections[:, 5], gt_cls, iou)
