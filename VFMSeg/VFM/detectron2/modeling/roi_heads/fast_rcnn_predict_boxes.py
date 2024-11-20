def predict_boxes(self, predictions: Tuple[torch.Tensor, torch.Tensor],
    proposals: List[Instances]):
    """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were
                used to compute predictions. The ``proposal_boxes`` field is expected.

        Returns:
            list[Tensor]:
                A list of Tensors of predicted class-specific or class-agnostic boxes
                for each image. Element i has shape (Ri, K * B) or (Ri, B), where Ri is
                the number of proposals for image i and B is the box dimension (4 or 5)
        """
    if not len(proposals):
        return []
    _, proposal_deltas = predictions
    num_prop_per_image = [len(p) for p in proposals]
    proposal_boxes = cat([p.proposal_boxes.tensor for p in proposals], dim=0)
    predict_boxes = self.box2box_transform.apply_deltas(proposal_deltas,
        proposal_boxes)
    return predict_boxes.split(num_prop_per_image)
