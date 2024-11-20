def _forward_mask(self, features: Dict[str, torch.Tensor], instances: List[
    Instances]):
    """
        Forward logic of the mask prediction branch.

        Args:
            features (dict[str, Tensor]): mapping from feature map names to tensor.
                Same as in :meth:`ROIHeads.forward`.
            instances (list[Instances]): the per-image instances to train/predict masks.
                In training, they can be the proposals.
                In inference, they can be the boxes predicted by R-CNN box head.

        Returns:
            In training, a dict of losses.
            In inference, update `instances` with new fields "pred_masks" and return it.
        """
    if not self.mask_on:
        return {} if self.training else instances
    if self.training:
        instances, _ = select_foreground_proposals(instances, self.num_classes)
    if self.mask_pooler is not None:
        features = [features[f] for f in self.mask_in_features]
        boxes = [(x.proposal_boxes if self.training else x.pred_boxes) for
            x in instances]
        features = self.mask_pooler(features, boxes)
    else:
        features = {f: features[f] for f in self.mask_in_features}
    return self.mask_head(features, instances)
