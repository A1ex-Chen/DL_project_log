def forward(self, images, features, proposals, targets=None):
    del images
    if self.training:
        proposals = self.label_and_sample_proposals(proposals, targets)
    if self.training:
        losses = self._forward_box(features, proposals, targets)
        losses.update(self._forward_mask(features, proposals))
        losses.update(self._forward_keypoint(features, proposals))
        return proposals, losses
    else:
        pred_instances = self._forward_box(features, proposals)
        pred_instances = self.forward_with_given_boxes(features, pred_instances
            )
        return pred_instances, {}
