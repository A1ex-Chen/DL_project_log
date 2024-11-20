def losses(self, predictions, proposals):
    """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were used
                to compute predictions. The fields ``proposal_boxes``, ``gt_boxes``,
                ``gt_classes`` are expected.

        Returns:
            Dict[str, Tensor]: dict of losses
        """
    scores, proposal_deltas = predictions
    gt_classes = cat([p.gt_classes for p in proposals], dim=0) if len(proposals
        ) else torch.empty(0)
    _log_classification_stats(scores, gt_classes)
    if len(proposals):
        proposal_boxes = cat([p.proposal_boxes.tensor for p in proposals],
            dim=0)
        assert not proposal_boxes.requires_grad, 'Proposals should not require gradients!'
        gt_boxes = cat([(p.gt_boxes if p.has('gt_boxes') else p.
            proposal_boxes).tensor for p in proposals], dim=0)
    else:
        proposal_boxes = gt_boxes = torch.empty((0, 4), device=
            proposal_deltas.device)
    losses = {'loss_cls': cross_entropy(scores, gt_classes, reduction=
        'mean'), 'loss_box_reg': self.box_reg_loss(proposal_boxes, gt_boxes,
        proposal_deltas, gt_classes)}
    return {k: (v * self.loss_weight.get(k, 1.0)) for k, v in losses.items()}
