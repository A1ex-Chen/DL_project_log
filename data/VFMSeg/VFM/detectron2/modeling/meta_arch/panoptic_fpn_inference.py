def inference(self, batched_inputs: List[Dict[str, torch.Tensor]],
    do_postprocess: bool=True):
    """
        Run inference on the given inputs.

        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            do_postprocess (bool): whether to apply post-processing on the outputs.

        Returns:
            When do_postprocess=True, see docs in :meth:`forward`.
            Otherwise, returns a (list[Instances], list[Tensor]) that contains
            the raw detector outputs, and raw semantic segmentation outputs.
        """
    images = self.preprocess_image(batched_inputs)
    features = self.backbone(images.tensor)
    sem_seg_results, sem_seg_losses = self.sem_seg_head(features, None)
    proposals, _ = self.proposal_generator(images, features, None)
    detector_results, _ = self.roi_heads(images, features, proposals, None)
    if do_postprocess:
        processed_results = []
        for sem_seg_result, detector_result, input_per_image, image_size in zip(
            sem_seg_results, detector_results, batched_inputs, images.
            image_sizes):
            height = input_per_image.get('height', image_size[0])
            width = input_per_image.get('width', image_size[1])
            sem_seg_r = sem_seg_postprocess(sem_seg_result, image_size,
                height, width)
            detector_r = detector_postprocess(detector_result, height, width)
            processed_results.append({'sem_seg': sem_seg_r, 'instances':
                detector_r})
            panoptic_r = combine_semantic_and_instance_outputs(detector_r,
                sem_seg_r.argmax(dim=0), self.combine_overlap_thresh, self.
                combine_stuff_area_thresh, self.combine_instances_score_thresh)
            processed_results[-1]['panoptic_seg'] = panoptic_r
        return processed_results
    else:
        return detector_results, sem_seg_results
