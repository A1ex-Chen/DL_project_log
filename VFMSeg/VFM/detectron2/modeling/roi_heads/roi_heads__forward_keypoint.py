def _forward_keypoint(self, features: Dict[str, torch.Tensor], instances:
    List[Instances]):
    """
        Forward logic of the keypoint prediction branch.

        Args:
            features (dict[str, Tensor]): mapping from feature map names to tensor.
                Same as in :meth:`ROIHeads.forward`.
            instances (list[Instances]): the per-image instances to train/predict keypoints.
                In training, they can be the proposals.
                In inference, they can be the boxes predicted by R-CNN box head.

        Returns:
            In training, a dict of losses.
            In inference, update `instances` with new fields "pred_keypoints" and return it.
        """
    if not self.keypoint_on:
        return {} if self.training else instances
    if self.training:
        instances, _ = select_foreground_proposals(instances, self.num_classes)
        instances = select_proposals_with_visible_keypoints(instances)
    if self.keypoint_pooler is not None:
        features = [features[f] for f in self.keypoint_in_features]
        boxes = [(x.proposal_boxes if self.training else x.pred_boxes) for
            x in instances]
        features = self.keypoint_pooler(features, boxes)
    else:
        features = {f: features[f] for f in self.keypoint_in_features}
    return self.keypoint_head(features, instances)
