def forward_with_given_boxes(self, features: Dict[str, torch.Tensor],
    instances: List[Instances]) ->List[Instances]:
    """
        Use the given boxes in `instances` to produce other (non-box) per-ROI outputs.

        This is useful for downstream tasks where a box is known, but need to obtain
        other attributes (outputs of other heads).
        Test-time augmentation also uses this.

        Args:
            features: same as in `forward()`
            instances (list[Instances]): instances to predict other outputs. Expect the keys
                "pred_boxes" and "pred_classes" to exist.

        Returns:
            list[Instances]:
                the same `Instances` objects, with extra
                fields such as `pred_masks` or `pred_keypoints`.
        """
    assert not self.training
    assert instances[0].has('pred_boxes') and instances[0].has('pred_classes')
    instances = self._forward_mask(features, instances)
    instances = self._forward_keypoint(features, instances)
    return instances
