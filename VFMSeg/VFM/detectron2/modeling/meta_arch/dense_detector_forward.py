def forward(self, batched_inputs: List[Dict[str, Tensor]]):
    """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances: Instances

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.

        Returns:
            In training, dict[str, Tensor]: mapping from a named loss to a tensor storing the
            loss. Used during training only. In inference, the standard output format, described
            in :doc:`/tutorials/models`.
        """
    images = self.preprocess_image(batched_inputs)
    features = self.backbone(images.tensor)
    features = [features[f] for f in self.head_in_features]
    predictions = self.head(features)
    if self.training:
        assert not torch.jit.is_scripting(), 'Not supported'
        assert 'instances' in batched_inputs[0
            ], 'Instance annotations are missing in training!'
        gt_instances = [x['instances'].to(self.device) for x in batched_inputs]
        return self.forward_training(images, features, predictions,
            gt_instances)
    else:
        results = self.forward_inference(images, features, predictions)
        if torch.jit.is_scripting():
            return results
        processed_results = []
        for results_per_image, input_per_image, image_size in zip(results,
            batched_inputs, images.image_sizes):
            height = input_per_image.get('height', image_size[0])
            width = input_per_image.get('width', image_size[1])
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({'instances': r})
        return processed_results
