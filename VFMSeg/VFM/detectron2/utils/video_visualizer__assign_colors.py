def _assign_colors(self, instances):
    """
        Naive tracking heuristics to assign same color to the same instance,
        will update the internal state of tracked instances.

        Returns:
            list[tuple[float]]: list of colors.
        """
    is_crowd = np.zeros((len(instances),), dtype=np.bool)
    if instances[0].bbox is None:
        assert instances[0].mask_rle is not None
        rles_old = [x.mask_rle for x in self._old_instances]
        rles_new = [x.mask_rle for x in instances]
        ious = mask_util.iou(rles_old, rles_new, is_crowd)
        threshold = 0.5
    else:
        boxes_old = [x.bbox for x in self._old_instances]
        boxes_new = [x.bbox for x in instances]
        ious = mask_util.iou(boxes_old, boxes_new, is_crowd)
        threshold = 0.6
    if len(ious) == 0:
        ious = np.zeros((len(self._old_instances), len(instances)), dtype=
            'float32')
    for old_idx, old in enumerate(self._old_instances):
        for new_idx, new in enumerate(instances):
            if old.label != new.label:
                ious[old_idx, new_idx] = 0
    matched_new_per_old = np.asarray(ious).argmax(axis=1)
    max_iou_per_old = np.asarray(ious).max(axis=1)
    extra_instances = []
    for idx, inst in enumerate(self._old_instances):
        if max_iou_per_old[idx] > threshold:
            newidx = matched_new_per_old[idx]
            if instances[newidx].color is None:
                instances[newidx].color = inst.color
                continue
        inst.ttl -= 1
        if inst.ttl > 0:
            extra_instances.append(inst)
    for inst in instances:
        if inst.color is None:
            inst.color = random_color(rgb=True, maximum=1)
    self._old_instances = instances[:] + extra_instances
    return [d.color for d in instances]
