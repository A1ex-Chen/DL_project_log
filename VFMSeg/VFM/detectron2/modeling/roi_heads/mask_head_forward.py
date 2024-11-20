def forward(self, x, instances: List[Instances]):
    """
        Args:
            x: input region feature(s) provided by :class:`ROIHeads`.
            instances (list[Instances]): contains the boxes & labels corresponding
                to the input features.
                Exact format is up to its caller to decide.
                Typically, this is the foreground instances in training, with
                "proposal_boxes" field and other gt annotations.
                In inference, it contains boxes that are already predicted.

        Returns:
            A dict of losses in training. The predicted "instances" in inference.
        """
    x = self.layers(x)
    if self.training:
        return {'loss_mask': mask_rcnn_loss(x, instances, self.vis_period) *
            self.loss_weight}
    else:
        mask_rcnn_inference(x, instances)
        return instances
