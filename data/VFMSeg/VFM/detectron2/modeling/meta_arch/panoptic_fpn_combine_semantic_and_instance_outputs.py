def combine_semantic_and_instance_outputs(instance_results,
    semantic_results, overlap_threshold, stuff_area_thresh,
    instances_score_thresh):
    """
    Implement a simple combining logic following
    "combine_semantic_and_instance_predictions.py" in panopticapi
    to produce panoptic segmentation outputs.

    Args:
        instance_results: output of :func:`detector_postprocess`.
        semantic_results: an (H, W) tensor, each element is the contiguous semantic
            category id

    Returns:
        panoptic_seg (Tensor): of shape (height, width) where the values are ids for each segment.
        segments_info (list[dict]): Describe each segment in `panoptic_seg`.
            Each dict contains keys "id", "category_id", "isthing".
    """
    panoptic_seg = torch.zeros_like(semantic_results, dtype=torch.int32)
    sorted_inds = torch.argsort(-instance_results.scores)
    current_segment_id = 0
    segments_info = []
    instance_masks = instance_results.pred_masks.to(dtype=torch.bool,
        device=panoptic_seg.device)
    for inst_id in sorted_inds:
        score = instance_results.scores[inst_id].item()
        if score < instances_score_thresh:
            break
        mask = instance_masks[inst_id]
        mask_area = mask.sum().item()
        if mask_area == 0:
            continue
        intersect = (mask > 0) & (panoptic_seg > 0)
        intersect_area = intersect.sum().item()
        if intersect_area * 1.0 / mask_area > overlap_threshold:
            continue
        if intersect_area > 0:
            mask = mask & (panoptic_seg == 0)
        current_segment_id += 1
        panoptic_seg[mask] = current_segment_id
        segments_info.append({'id': current_segment_id, 'isthing': True,
            'score': score, 'category_id': instance_results.pred_classes[
            inst_id].item(), 'instance_id': inst_id.item()})
    semantic_labels = torch.unique(semantic_results).cpu().tolist()
    for semantic_label in semantic_labels:
        if semantic_label == 0:
            continue
        mask = (semantic_results == semantic_label) & (panoptic_seg == 0)
        mask_area = mask.sum().item()
        if mask_area < stuff_area_thresh:
            continue
        current_segment_id += 1
        panoptic_seg[mask] = current_segment_id
        segments_info.append({'id': current_segment_id, 'isthing': False,
            'category_id': semantic_label, 'area': mask_area})
    return panoptic_seg, segments_info
