def _run_stage(self, features, proposals, stage):
    """
        Args:
            features (list[Tensor]): #lvl input features to ROIHeads
            proposals (list[Instances]): #image Instances, with the field "proposal_boxes"
            stage (int): the current stage

        Returns:
            Same output as `FastRCNNOutputLayers.forward()`.
        """
    box_features = self.box_pooler(features, [x.proposal_boxes for x in
        proposals])
    if self.training:
        box_features = _ScaleGradient.apply(box_features, 1.0 / self.
            num_cascade_stages)
    box_features = self.box_head[stage](box_features)
    return self.box_predictor[stage](box_features)
