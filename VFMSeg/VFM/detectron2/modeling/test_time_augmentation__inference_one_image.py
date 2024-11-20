def _inference_one_image(self, input):
    """
        Args:
            input (dict): one dataset dict with "image" field being a CHW tensor

        Returns:
            dict: one output dict
        """
    orig_shape = input['height'], input['width']
    augmented_inputs, tfms = self._get_augmented_inputs(input)
    with self._turn_off_roi_heads(['mask_on', 'keypoint_on']):
        all_boxes, all_scores, all_classes = self._get_augmented_boxes(
            augmented_inputs, tfms)
    merged_instances = self._merge_detections(all_boxes, all_scores,
        all_classes, orig_shape)
    if self.cfg.MODEL.MASK_ON:
        augmented_instances = self._rescale_detected_boxes(augmented_inputs,
            merged_instances, tfms)
        outputs = self._batch_inference(augmented_inputs, augmented_instances)
        del augmented_inputs, augmented_instances
        merged_instances.pred_masks = self._reduce_pred_masks(outputs, tfms)
        merged_instances = detector_postprocess(merged_instances, *orig_shape)
        return {'instances': merged_instances}
    else:
        return {'instances': merged_instances}
