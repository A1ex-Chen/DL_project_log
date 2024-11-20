def draw_panoptic_seg_predictions(self, frame, panoptic_seg, segments_info,
    area_threshold=None, alpha=0.5):
    frame_visualizer = Visualizer(frame, self.metadata)
    pred = _PanopticPrediction(panoptic_seg, segments_info, self.metadata)
    if self._instance_mode == ColorMode.IMAGE_BW:
        frame_visualizer.output.reset_image(frame_visualizer.
            _create_grayscale_image(pred.non_empty_mask()))
    for mask, sinfo in pred.semantic_masks():
        category_idx = sinfo['category_id']
        try:
            mask_color = [(x / 255) for x in self.metadata.stuff_colors[
                category_idx]]
        except AttributeError:
            mask_color = None
        frame_visualizer.draw_binary_mask(mask, color=mask_color, text=self
            .metadata.stuff_classes[category_idx], alpha=alpha,
            area_threshold=area_threshold)
    all_instances = list(pred.instance_masks())
    if len(all_instances) == 0:
        return frame_visualizer.output
    masks, sinfo = list(zip(*all_instances))
    num_instances = len(masks)
    masks_rles = mask_util.encode(np.asarray(np.asarray(masks).transpose(1,
        2, 0), dtype=np.uint8, order='F'))
    assert len(masks_rles) == num_instances
    category_ids = [x['category_id'] for x in sinfo]
    detected = [_DetectedInstance(category_ids[i], bbox=None, mask_rle=
        masks_rles[i], color=None, ttl=8) for i in range(num_instances)]
    colors = self._assign_colors(detected)
    labels = [self.metadata.thing_classes[k] for k in category_ids]
    frame_visualizer.overlay_instances(boxes=None, masks=masks, labels=
        labels, keypoints=None, assigned_colors=colors, alpha=alpha)
    return frame_visualizer.output
