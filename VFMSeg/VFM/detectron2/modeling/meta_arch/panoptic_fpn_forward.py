def forward(self, batched_inputs):
    """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.

                For now, each item in the list is a dict that contains:

                * "image": Tensor, image in (C, H, W) format.
                * "instances": Instances
                * "sem_seg": semantic segmentation ground truth.
                * Other information that's included in the original dicts, such as:
                  "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.

        Returns:
            list[dict]:
                each dict has the results for one image. The dict contains the following keys:

                * "instances": see :meth:`GeneralizedRCNN.forward` for its format.
                * "sem_seg": see :meth:`SemanticSegmentor.forward` for its format.
                * "panoptic_seg": See the return value of
                  :func:`combine_semantic_and_instance_outputs` for its format.
        """
    if not self.training:
        return self.inference(batched_inputs)
    images = self.preprocess_image(batched_inputs)
    features = self.backbone(images.tensor)
    assert 'sem_seg' in batched_inputs[0]
    gt_sem_seg = [x['sem_seg'].to(self.device) for x in batched_inputs]
    gt_sem_seg = ImageList.from_tensors(gt_sem_seg, self.backbone.
        size_divisibility, self.sem_seg_head.ignore_value).tensor
    sem_seg_results, sem_seg_losses = self.sem_seg_head(features, gt_sem_seg)
    gt_instances = [x['instances'].to(self.device) for x in batched_inputs]
    proposals, proposal_losses = self.proposal_generator(images, features,
        gt_instances)
    detector_results, detector_losses = self.roi_heads(images, features,
        proposals, gt_instances)
    losses = sem_seg_losses
    losses.update(proposal_losses)
    losses.update(detector_losses)
    return losses
