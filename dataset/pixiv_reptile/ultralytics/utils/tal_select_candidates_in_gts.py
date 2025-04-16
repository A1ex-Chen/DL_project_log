@staticmethod
def select_candidates_in_gts(xy_centers, gt_bboxes):
    """
        Select the positive anchor center in gt for rotated bounding boxes.

        Args:
            xy_centers (Tensor): shape(h*w, 2)
            gt_bboxes (Tensor): shape(b, n_boxes, 5)

        Returns:
            (Tensor): shape(b, n_boxes, h*w)
        """
    corners = xywhr2xyxyxyxy(gt_bboxes)
    a, b, _, d = corners.split(1, dim=-2)
    ab = b - a
    ad = d - a
    ap = xy_centers - a
    norm_ab = (ab * ab).sum(dim=-1)
    norm_ad = (ad * ad).sum(dim=-1)
    ap_dot_ab = (ap * ab).sum(dim=-1)
    ap_dot_ad = (ap * ad).sum(dim=-1)
    return (ap_dot_ab >= 0) & (ap_dot_ab <= norm_ab) & (ap_dot_ad >= 0) & (
        ap_dot_ad <= norm_ad)
