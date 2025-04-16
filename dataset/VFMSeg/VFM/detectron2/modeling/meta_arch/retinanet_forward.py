def forward(self, features: List[Tensor]):
    """
        Arguments:
            features (list[Tensor]): FPN feature map tensors in high to low resolution.
                Each tensor in the list correspond to different feature levels.

        Returns:
            logits (list[Tensor]): #lvl tensors, each has shape (N, AxK, Hi, Wi).
                The tensor predicts the classification probability
                at each spatial position for each of the A anchors and K object
                classes.
            bbox_reg (list[Tensor]): #lvl tensors, each has shape (N, Ax4, Hi, Wi).
                The tensor predicts 4-vector (dx,dy,dw,dh) box
                regression values for every anchor. These values are the
                relative offset between the anchor and the ground truth box.
        """
    assert len(features) == self._num_features
    logits = []
    bbox_reg = []
    for feature in features:
        logits.append(self.cls_score(self.cls_subnet(feature)))
        bbox_reg.append(self.bbox_pred(self.bbox_subnet(feature)))
    return logits, bbox_reg
