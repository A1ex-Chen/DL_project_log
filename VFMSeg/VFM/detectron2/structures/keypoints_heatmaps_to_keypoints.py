@torch.jit.script_if_tracing
def heatmaps_to_keypoints(maps: torch.Tensor, rois: torch.Tensor
    ) ->torch.Tensor:
    """
    Extract predicted keypoint locations from heatmaps.

    Args:
        maps (Tensor): (#ROIs, #keypoints, POOL_H, POOL_W). The predicted heatmap of logits for
            each ROI and each keypoint.
        rois (Tensor): (#ROIs, 4). The box of each ROI.

    Returns:
        Tensor of shape (#ROIs, #keypoints, 4) with the last dimension corresponding to
        (x, y, logit, score) for each keypoint.

    When converting discrete pixel indices in an NxN image to a continuous keypoint coordinate,
    we maintain consistency with :meth:`Keypoints.to_heatmap` by using the conversion from
    Heckbert 1990: c = d + 0.5, where d is a discrete coordinate and c is a continuous coordinate.
    """
    maps = maps.detach()
    rois = rois.detach()
    offset_x = rois[:, 0]
    offset_y = rois[:, 1]
    widths = (rois[:, 2] - rois[:, 0]).clamp(min=1)
    heights = (rois[:, 3] - rois[:, 1]).clamp(min=1)
    widths_ceil = widths.ceil()
    heights_ceil = heights.ceil()
    num_rois, num_keypoints = maps.shape[:2]
    xy_preds = maps.new_zeros(rois.shape[0], num_keypoints, 4)
    width_corrections = widths / widths_ceil
    height_corrections = heights / heights_ceil
    keypoints_idx = torch.arange(num_keypoints, device=maps.device)
    for i in range(num_rois):
        outsize = int(heights_ceil[i]), int(widths_ceil[i])
        roi_map = F.interpolate(maps[[i]], size=outsize, mode='bicubic',
            align_corners=False).squeeze(0)
        max_score, _ = roi_map.view(num_keypoints, -1).max(1)
        max_score = max_score.view(num_keypoints, 1, 1)
        tmp_full_resolution = (roi_map - max_score).exp_()
        tmp_pool_resolution = (maps[i] - max_score).exp_()
        roi_map_scores = tmp_full_resolution / tmp_pool_resolution.sum((1, 
            2), keepdim=True)
        w = roi_map.shape[2]
        pos = roi_map.view(num_keypoints, -1).argmax(1)
        x_int = pos % w
        y_int = (pos - x_int) // w
        assert (roi_map_scores[keypoints_idx, y_int, x_int] ==
            roi_map_scores.view(num_keypoints, -1).max(1)[0]).all()
        x = (x_int.float() + 0.5) * width_corrections[i]
        y = (y_int.float() + 0.5) * height_corrections[i]
        xy_preds[i, :, 0] = x + offset_x[i]
        xy_preds[i, :, 1] = y + offset_y[i]
        xy_preds[i, :, 2] = roi_map[keypoints_idx, y_int, x_int]
        xy_preds[i, :, 3] = roi_map_scores[keypoints_idx, y_int, x_int]
    return xy_preds
