def _get_loss_aux(self, pred_bboxes, pred_scores, gt_bboxes, gt_cls,
    gt_groups, match_indices=None, postfix='', masks=None, gt_mask=None):
    """Get auxiliary losses."""
    loss = torch.zeros(5 if masks is not None else 3, device=pred_bboxes.device
        )
    if match_indices is None and self.use_uni_match:
        match_indices = self.matcher(pred_bboxes[self.uni_match_ind],
            pred_scores[self.uni_match_ind], gt_bboxes, gt_cls, gt_groups,
            masks=masks[self.uni_match_ind] if masks is not None else None,
            gt_mask=gt_mask)
    for i, (aux_bboxes, aux_scores) in enumerate(zip(pred_bboxes, pred_scores)
        ):
        aux_masks = masks[i] if masks is not None else None
        loss_ = self._get_loss(aux_bboxes, aux_scores, gt_bboxes, gt_cls,
            gt_groups, masks=aux_masks, gt_mask=gt_mask, postfix=postfix,
            match_indices=match_indices)
        loss[0] += loss_[f'loss_class{postfix}']
        loss[1] += loss_[f'loss_bbox{postfix}']
        loss[2] += loss_[f'loss_giou{postfix}']
    loss = {f'loss_class_aux{postfix}': loss[0], f'loss_bbox_aux{postfix}':
        loss[1], f'loss_giou_aux{postfix}': loss[2]}
    return loss
