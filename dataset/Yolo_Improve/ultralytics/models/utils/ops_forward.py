def forward(self, pred_bboxes, pred_scores, gt_bboxes, gt_cls, gt_groups,
    masks=None, gt_mask=None):
    """
        Forward pass for HungarianMatcher. This function computes costs based on prediction and ground truth
        (classification cost, L1 cost between boxes and GIoU cost between boxes) and finds the optimal matching between
        predictions and ground truth based on these costs.

        Args:
            pred_bboxes (Tensor): Predicted bounding boxes with shape [batch_size, num_queries, 4].
            pred_scores (Tensor): Predicted scores with shape [batch_size, num_queries, num_classes].
            gt_cls (torch.Tensor): Ground truth classes with shape [num_gts, ].
            gt_bboxes (torch.Tensor): Ground truth bounding boxes with shape [num_gts, 4].
            gt_groups (List[int]): List of length equal to batch size, containing the number of ground truths for
                each image.
            masks (Tensor, optional): Predicted masks with shape [batch_size, num_queries, height, width].
                Defaults to None.
            gt_mask (List[Tensor], optional): List of ground truth masks, each with shape [num_masks, Height, Width].
                Defaults to None.

        Returns:
            (List[Tuple[Tensor, Tensor]]): A list of size batch_size, each element is a tuple (index_i, index_j), where:
                - index_i is the tensor of indices of the selected predictions (in order)
                - index_j is the tensor of indices of the corresponding selected ground truth targets (in order)
                For each batch element, it holds:
                    len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
    bs, nq, nc = pred_scores.shape
    if sum(gt_groups) == 0:
        return [(torch.tensor([], dtype=torch.long), torch.tensor([], dtype
            =torch.long)) for _ in range(bs)]
    pred_scores = pred_scores.detach().view(-1, nc)
    pred_scores = F.sigmoid(pred_scores) if self.use_fl else F.softmax(
        pred_scores, dim=-1)
    pred_bboxes = pred_bboxes.detach().view(-1, 4)
    pred_scores = pred_scores[:, gt_cls]
    if self.use_fl:
        neg_cost_class = (1 - self.alpha) * pred_scores ** self.gamma * -(1 -
            pred_scores + 1e-08).log()
        pos_cost_class = self.alpha * (1 - pred_scores) ** self.gamma * -(
            pred_scores + 1e-08).log()
        cost_class = pos_cost_class - neg_cost_class
    else:
        cost_class = -pred_scores
    cost_bbox = (pred_bboxes.unsqueeze(1) - gt_bboxes.unsqueeze(0)).abs().sum(
        -1)
    cost_giou = 1.0 - bbox_iou(pred_bboxes.unsqueeze(1), gt_bboxes.
        unsqueeze(0), xywh=True, GIoU=True).squeeze(-1)
    C = self.cost_gain['class'] * cost_class + self.cost_gain['bbox'
        ] * cost_bbox + self.cost_gain['giou'] * cost_giou
    if self.with_mask:
        C += self._cost_mask(bs, gt_groups, masks, gt_mask)
    C[C.isnan() | C.isinf()] = 0.0
    C = C.view(bs, nq, -1).cpu()
    indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(
        gt_groups, -1))]
    gt_groups = torch.as_tensor([0, *gt_groups[:-1]]).cumsum_(0)
    return [(torch.tensor(i, dtype=torch.long), torch.tensor(j, dtype=torch
        .long) + gt_groups[k]) for k, (i, j) in enumerate(indices)]
