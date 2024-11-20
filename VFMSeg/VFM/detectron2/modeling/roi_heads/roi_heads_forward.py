def forward(self, images: ImageList, features: Dict[str, torch.Tensor],
    proposals: List[Instances], targets: Optional[List[Instances]]=None
    ) ->Tuple[List[Instances], Dict[str, torch.Tensor]]:
    """
        See :class:`ROIHeads.forward`.
        """
    del images
    if self.training:
        assert targets, "'targets' argument is required during training"
        proposals = self.label_and_sample_proposals(proposals, targets)
    del targets
    if self.training:
        losses = self._forward_box(features, proposals)
        losses.update(self._forward_mask(features, proposals))
        losses.update(self._forward_keypoint(features, proposals))
        return proposals, losses
    else:
        pred_instances = self._forward_box(features, proposals)
        pred_instances = self.forward_with_given_boxes(features, pred_instances
            )
        return pred_instances, {}
