@torch.no_grad()
def _match_anchors(self, gt_boxes: Boxes, anchors: List[Boxes]):
    """
        Match ground-truth boxes to a set of multi-level anchors.

        Args:
            gt_boxes: Ground-truth boxes from instances of an image.
            anchors: List of anchors for each feature map (of different scales).

        Returns:
            torch.Tensor
                A tensor of shape `(M, R)`, given `M` ground-truth boxes and total
                `R` anchor points from all feature levels, indicating the quality
                of match between m-th box and r-th anchor. Higher value indicates
                better match.
        """
    num_anchors_per_level = [len(x) for x in anchors]
    anchors = Boxes.cat(anchors)
    anchor_centers = anchors.get_centers()
    anchor_sizes = anchors.tensor[:, 2] - anchors.tensor[:, 0]
    lower_bound = anchor_sizes * 4
    lower_bound[:num_anchors_per_level[0]] = 0
    upper_bound = anchor_sizes * 8
    upper_bound[-num_anchors_per_level[-1]:] = float('inf')
    gt_centers = gt_boxes.get_centers()
    center_dists = (anchor_centers[None, :, :] - gt_centers[:, None, :]).abs_()
    sampling_regions = self.center_sampling_radius * anchor_sizes[None, :]
    match_quality_matrix = center_dists.max(dim=2).values < sampling_regions
    pairwise_dist = pairwise_point_box_distance(anchor_centers, gt_boxes)
    pairwise_dist = pairwise_dist.permute(1, 0, 2)
    match_quality_matrix &= pairwise_dist.min(dim=2).values > 0
    pairwise_dist = pairwise_dist.max(dim=2).values
    match_quality_matrix &= (pairwise_dist > lower_bound[None, :]) & (
        pairwise_dist < upper_bound[None, :])
    gt_areas = gt_boxes.area()
    match_quality_matrix = match_quality_matrix.to(torch.float32)
    match_quality_matrix *= 100000000.0 - gt_areas[:, None]
    return match_quality_matrix
