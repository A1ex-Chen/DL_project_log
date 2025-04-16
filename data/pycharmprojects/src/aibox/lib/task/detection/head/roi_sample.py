def sample(self, proposal_bboxes: Tensor, gt_bboxes: Tensor, gt_classes: Tensor
    ) ->Tuple[Tensor, Tensor, Tensor]:
    sampled_indices = torch.arange(proposal_bboxes.shape[0]).to(proposal_bboxes
        .device)
    labels = torch.full((proposal_bboxes.shape[0],), -1, dtype=torch.long,
        device=proposal_bboxes.device)
    ious = box_iou(proposal_bboxes, gt_bboxes)
    proposal_max_ious, proposal_assignments = ious.max(dim=1)
    gt_max_ious, gt_assignments = ious.max(dim=0)
    low_quality_indices = (proposal_max_ious < 0.5).nonzero().flatten()
    addition_indices = ((ious >= 0.3) & (ious == gt_max_ious.unsqueeze(dim=0))
        ).nonzero()[:, 0]
    addition_gt_classes = gt_classes[proposal_assignments[addition_indices]]
    high_quality_indices = (proposal_max_ious >= 0.5).nonzero().flatten()
    high_quality_gt_classes = gt_classes[proposal_assignments[
        high_quality_indices]]
    labels[low_quality_indices] = 0
    labels[addition_indices] = addition_gt_classes
    labels[high_quality_indices] = high_quality_gt_classes
    fg_indices = (labels > 0).nonzero().flatten()
    bg_indices = (labels == 0).nonzero().flatten()
    explicit_bg_indices = torch.cat([addition_indices[addition_gt_classes ==
        0], high_quality_indices[high_quality_gt_classes == 0]], dim=0)
    expected_num_fg_indices = int(self._num_proposal_samples_per_batch * 0.5)
    fg_indices = fg_indices[torch.randperm(fg_indices.shape[0])[:
        expected_num_fg_indices]]
    expected_num_bg_indices = (self._num_proposal_samples_per_batch -
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
    proposal_bboxes = proposal_bboxes[selected_indices]
    sampled_indices = sampled_indices[selected_indices]
    gt_bboxes = gt_bboxes[proposal_assignments[selected_indices]]
    gt_proposal_classes = labels[selected_indices]
    gt_proposal_transformers = BBox.calc_transformer(proposal_bboxes, gt_bboxes
        )
    return sampled_indices, gt_proposal_classes, gt_proposal_transformers
