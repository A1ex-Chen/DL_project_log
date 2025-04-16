def forward(self, batched_inputs):
    """
        Args:
            Same as in :class:`GeneralizedRCNN.forward`

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "proposals" whose value is a
                :class:`Instances` with keys "proposal_boxes" and "objectness_logits".
        """
    images = [x['image'].to(self.device) for x in batched_inputs]
    images = [((x - self.pixel_mean) / self.pixel_std) for x in images]
    images = ImageList.from_tensors(images, self.backbone.size_divisibility)
    features = self.backbone(images.tensor)
    if 'instances' in batched_inputs[0]:
        gt_instances = [x['instances'].to(self.device) for x in batched_inputs]
    elif 'targets' in batched_inputs[0]:
        log_first_n(logging.WARN,
            "'targets' in the model inputs is now renamed to 'instances'!",
            n=10)
        gt_instances = [x['targets'].to(self.device) for x in batched_inputs]
    else:
        gt_instances = None
    proposals, proposal_losses = self.proposal_generator(images, features,
        gt_instances)
    if self.training:
        return proposal_losses
    processed_results = []
    for results_per_image, input_per_image, image_size in zip(proposals,
        batched_inputs, images.image_sizes):
        height = input_per_image.get('height', image_size[0])
        width = input_per_image.get('width', image_size[1])
        r = detector_postprocess(results_per_image, height, width)
        processed_results.append({'proposals': r})
    return processed_results
