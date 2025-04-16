def forward(self, x, instances: List[Instances]):
    """
        Args:
            x: input 4D region feature(s) provided by :class:`ROIHeads`.
            instances (list[Instances]): contains the boxes & labels corresponding
                to the input features.
                Exact format is up to its caller to decide.
                Typically, this is the foreground instances in training, with
                "proposal_boxes" field and other gt annotations.
                In inference, it contains boxes that are already predicted.

        Returns:
            A dict of losses if in training. The predicted "instances" if in inference.
        """
    x = self.layers(x)
    if self.training:
        num_images = len(instances)
        normalizer = (None if self.loss_normalizer == 'visible' else 
            num_images * self.loss_normalizer)
        return {'loss_keypoint': keypoint_rcnn_loss(x, instances,
            normalizer=normalizer) * self.loss_weight}
    else:
        keypoint_rcnn_inference(x, instances)
        return instances
