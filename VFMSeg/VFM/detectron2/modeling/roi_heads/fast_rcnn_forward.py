def forward(self, x):
    """
        Args:
            x: per-region features of shape (N, ...) for N bounding boxes to predict.

        Returns:
            (Tensor, Tensor):
            First tensor: shape (N,K+1), scores for each of the N box. Each row contains the
            scores for K object categories and 1 background class.

            Second tensor: bounding box regression deltas for each box. Shape is shape (N,Kx4),
            or (N,4) for class-agnostic regression.
        """
    if x.dim() > 2:
        x = torch.flatten(x, start_dim=1)
    scores = self.cls_score(x)
    proposal_deltas = self.bbox_pred(x)
    return scores, proposal_deltas
