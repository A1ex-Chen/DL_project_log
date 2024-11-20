def bbox_decode(self, anchor_points, pred_dist, pred_angle):
    """
        Decode predicted object bounding box coordinates from anchor points and distribution.

        Args:
            anchor_points (torch.Tensor): Anchor points, (h*w, 2).
            pred_dist (torch.Tensor): Predicted rotated distance, (bs, h*w, 4).
            pred_angle (torch.Tensor): Predicted angle, (bs, h*w, 1).

        Returns:
            (torch.Tensor): Predicted rotated bounding boxes with angles, (bs, h*w, 5).
        """
    if self.use_dfl:
        b, a, c = pred_dist.shape
        pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.
            proj.type(pred_dist.dtype))
    return torch.cat((dist2rbox(pred_dist, pred_angle, anchor_points),
        pred_angle), dim=-1)
