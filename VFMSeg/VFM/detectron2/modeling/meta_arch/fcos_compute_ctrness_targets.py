def compute_ctrness_targets(self, anchors: List[Boxes], gt_boxes: List[
    torch.Tensor]):
    anchors = Boxes.cat(anchors).tensor
    reg_targets = [self.box2box_transform.get_deltas(anchors, m) for m in
        gt_boxes]
    reg_targets = torch.stack(reg_targets, dim=0)
    if len(reg_targets) == 0:
        return reg_targets.new_zeros(len(reg_targets))
    left_right = reg_targets[:, :, [0, 2]]
    top_bottom = reg_targets[:, :, [1, 3]]
    ctrness = left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0] * (
        top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
    return torch.sqrt(ctrness)
