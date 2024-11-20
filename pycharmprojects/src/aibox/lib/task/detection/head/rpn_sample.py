def sample(self, anchor_bboxes: Tensor, gt_bboxes: Tensor, gt_classes:
    Tensor, padded_image_width: int, padded_image_height: int) ->Tuple[
    Tensor, Optional[Tensor], Optional[Tensor]]:
    sampled_indices = torch.arange(anchor_bboxes.shape[0]).to(anchor_bboxes
        .device)
    inside_indices = BBox.inside(anchor_bboxes, left=0, top=0, right=
        padded_image_width, bottom=padded_image_height).nonzero().flatten()
    anchor_bboxes = anchor_bboxes[inside_indices]
    sampled_indices = sampled_indices[inside_indices]
    if sampled_indices.shape[0] == 0:
        return sampled_indices, None, None
    labels = torch.full((anchor_bboxes.shape[0],), -1, dtype=torch.long,
        device=anchor_bboxes.device)
    ious = box_iou(anchor_bboxes, gt_bboxes)
    anchor_max_ious, anchor_assignments = ious.max(dim=1)
    gt_max_ious, gt_assignments = ious.max(dim=0)
    low_quality_indices = (anchor_max_ious < 0.3).nonzero().flatten()
    addition_indices = ((ious >= 0.1) & (ious == gt_max_ious.unsqueeze(dim=0))
        ).nonzero()[:, 0]
    addition_gt_classes = gt_classes[anchor_assignments[addition_indices]]
    high_quality_indices = (anchor_max_ious >= 0.7).nonzero().flatten()
    high_quality_gt_classes = gt_classes[anchor_assignments[
        high_quality_indices]]
    labels[low_quality_indices] = 0
    labels[addition_indices] = addition_gt_classes
    labels[high_quality_indices] = high_quality_gt_classes
    fg_indices = (labels > 0).nonzero().flatten()
    bg_indices = (labels == 0).nonzero().flatten()
    explicit_bg_indices = torch.cat([addition_indices[addition_gt_classes ==
        0], high_quality_indices[high_quality_gt_classes == 0]], dim=0)
    expected_num_fg_indices = int(self._num_anchor_samples_per_batch * 0.5)
    fg_indices = fg_indices[torch.randperm(fg_indices.shape[0])[:
        expected_num_fg_indices]]
    expected_num_bg_indices = (self._num_anchor_samples_per_batch -
        fg_indices.shape[0])
    explicit_bg_indices = explicit_bg_indices[torch.randperm(
        explicit_bg_indices.shape[0])][:expected_num_bg_indices // 2]
    bg_indices = torch.cat([bg_indices[torch.randperm(bg_indices.shape[0])[
        :expected_num_bg_indices - explicit_bg_indices.shape[0]]],
        explicit_bg_indices], dim=0).unique(dim=0)
    bg_indices = bg_indices[torch.randperm(bg_indices.shape[0])]
    bg_fg_max_ratio = 5
    bg_indices = bg_indices[:fg_indices.shape[0] * bg_fg_max_ratio + 1]
    selected_indices = torch.cat([fg_indices, bg_indices], dim=0)
    selected_indices = selected_indices[torch.randperm(selected_indices.
        shape[0])]
    anchor_bboxes = anchor_bboxes[selected_indices]
    sampled_indices = sampled_indices[selected_indices]
    gt_bboxes = gt_bboxes[anchor_assignments[selected_indices]]
    gt_anchor_objectnesses = labels[selected_indices].gt(0).long()
    gt_anchor_transformers = BBox.calc_transformer(anchor_bboxes, gt_bboxes)
    return sampled_indices, gt_anchor_objectnesses, gt_anchor_transformers
