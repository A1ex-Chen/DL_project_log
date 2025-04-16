def _decode_proposals(self, anchors: List[Boxes], pred_anchor_deltas: List[
    torch.Tensor]):
    """
        Transform anchors into proposals by applying the predicted anchor deltas.

        Returns:
            proposals (list[Tensor]): A list of L tensors. Tensor i has shape
                (N, Hi*Wi*A, B)
        """
    N = pred_anchor_deltas[0].shape[0]
    proposals = []
    for anchors_i, pred_anchor_deltas_i in zip(anchors, pred_anchor_deltas):
        B = anchors_i.tensor.size(1)
        pred_anchor_deltas_i = pred_anchor_deltas_i.reshape(-1, B)
        anchors_i = anchors_i.tensor.unsqueeze(0).expand(N, -1, -1).reshape(
            -1, B)
        proposals_i = self.box2box_transform.apply_deltas(pred_anchor_deltas_i,
            anchors_i)
        proposals.append(proposals_i.view(N, -1, B))
    return proposals
