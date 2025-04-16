def _forward_box(self, features: Dict[str, torch.Tensor], proposals: List[
    Instances]):
    """
        Forward logic of the box prediction branch. If `self.train_on_pred_boxes is True`,
            the function puts predicted boxes in the `proposal_boxes` field of `proposals` argument.

        Args:
            features (dict[str, Tensor]): mapping from feature map names to tensor.
                Same as in :meth:`ROIHeads.forward`.
            proposals (list[Instances]): the per-image object proposals with
                their matching ground truth.
                Each has fields "proposal_boxes", and "objectness_logits",
                "gt_classes", "gt_boxes".

        Returns:
            In training, a dict of losses.
            In inference, a list of `Instances`, the predicted instances.
        """
    features = [features[f] for f in self.box_in_features]
    box_features = self.box_pooler(features, [x.proposal_boxes for x in
        proposals])
    box_features = self.box_head(box_features)
    predictions = self.box_predictor(box_features)
    del box_features
    if self.training:
        losses = self.box_predictor.losses(predictions, proposals)
        if self.train_on_pred_boxes:
            with torch.no_grad():
                pred_boxes = self.box_predictor.predict_boxes_for_gt_classes(
                    predictions, proposals)
                for proposals_per_image, pred_boxes_per_image in zip(proposals,
                    pred_boxes):
                    proposals_per_image.proposal_boxes = Boxes(
                        pred_boxes_per_image)
        return losses
    else:
        pred_instances, _ = self.box_predictor.inference(predictions, proposals
            )
        return pred_instances
