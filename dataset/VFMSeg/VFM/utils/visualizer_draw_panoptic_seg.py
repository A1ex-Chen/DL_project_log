def draw_panoptic_seg(self, panoptic_seg, segments_info, area_threshold=
    None, alpha=0.7):
    """
        Draw panoptic prediction annotations or results.

        Args:
            panoptic_seg (Tensor): of shape (height, width) where the values are ids for each
                segment.
            segments_info (list[dict] or None): Describe each segment in `panoptic_seg`.
                If it is a ``list[dict]``, each dict contains keys "id", "category_id".
                If None, category id of each pixel is computed by
                ``pixel // metadata.label_divisor``.
            area_threshold (int): stuff segments with less than `area_threshold` are not drawn.

        Returns:
            output (VisImage): image object with visualizations.
        """
    pred = _PanopticPrediction(panoptic_seg, segments_info, self.metadata)
    if self._instance_mode == ColorMode.IMAGE_BW:
        self.output.reset_image(self._create_grayscale_image(pred.
            non_empty_mask()))
    for mask, sinfo in pred.semantic_masks():
        category_idx = sinfo['category_id']
        try:
            mask_color = [(x / 255) for x in self.metadata.stuff_colors[
                category_idx]]
        except AttributeError:
            mask_color = None
        text = self.metadata.stuff_classes[category_idx].replace('-other', ''
            ).replace('-merged', '')
        self.draw_binary_mask(mask, color=mask_color, edge_color=_OFF_WHITE,
            text=text, alpha=alpha, area_threshold=area_threshold)
    all_instances = list(pred.instance_masks())
    if len(all_instances) == 0:
        return self.output
    masks, sinfo = list(zip(*all_instances))
    category_ids = [x['category_id'] for x in sinfo]
    try:
        scores = [x['score'] for x in sinfo]
    except KeyError:
        scores = None
    class_names = [name.replace('-other', '').replace('-merged', '') for
        name in self.metadata.thing_classes]
    labels = _create_text_labels(category_ids, scores, class_names, [x.get(
        'iscrowd', 0) for x in sinfo])
    try:
        colors = [self._jitter([(x / 255) for x in self.metadata.
            thing_colors[c]]) for c in category_ids]
    except AttributeError:
        colors = None
    self.overlay_instances(masks=masks, labels=labels, assigned_colors=
        colors, alpha=alpha)
    return self.output
