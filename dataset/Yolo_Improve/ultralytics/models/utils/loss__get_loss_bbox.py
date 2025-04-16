def _get_loss_bbox(self, pred_bboxes, gt_bboxes, postfix=''):
    """Calculates and returns the bounding box loss and GIoU loss for the predicted and ground truth bounding
        boxes.
        """
    name_bbox = f'loss_bbox{postfix}'
    name_giou = f'loss_giou{postfix}'
    loss = {}
    if len(gt_bboxes) == 0:
        loss[name_bbox] = torch.tensor(0.0, device=self.device)
        loss[name_giou] = torch.tensor(0.0, device=self.device)
        return loss
    loss[name_bbox] = self.loss_gain['bbox'] * F.l1_loss(pred_bboxes,
        gt_bboxes, reduction='sum') / len(gt_bboxes)
    loss[name_giou] = 1.0 - bbox_iou(pred_bboxes, gt_bboxes, xywh=True,
        GIoU=True)
    loss[name_giou] = loss[name_giou].sum() / len(gt_bboxes)
    loss[name_giou] = self.loss_gain['giou'] * loss[name_giou]
    return {k: v.squeeze() for k, v in loss.items()}
