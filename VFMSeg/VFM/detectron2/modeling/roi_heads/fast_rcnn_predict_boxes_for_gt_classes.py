def predict_boxes_for_gt_classes(self, predictions, proposals):
    """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were used
                to compute predictions. The fields ``proposal_boxes``, ``gt_classes`` are expected.

        Returns:
            list[Tensor]:
                A list of Tensors of predicted boxes for GT classes in case of
                class-specific box head. Element i of the list has shape (Ri, B), where Ri is
                the number of proposals for image i and B is the box dimension (4 or 5)
        """
    if not len(proposals):
        return []
    scores, proposal_deltas = predictions
    proposal_boxes = cat([p.proposal_boxes.tensor for p in proposals], dim=0)
    N, B = proposal_boxes.shape
    predict_boxes = self.box2box_transform.apply_deltas(proposal_deltas,
        proposal_boxes)
    K = predict_boxes.shape[1] // B
    if K > 1:
        gt_classes = torch.cat([p.gt_classes for p in proposals], dim=0)
        gt_classes = gt_classes.clamp_(0, K - 1)
        predict_boxes = predict_boxes.view(N, K, B)[torch.arange(N, dtype=
            torch.long, device=predict_boxes.device), gt_classes]
    num_prop_per_image = [len(p) for p in proposals]
    return predict_boxes.split(num_prop_per_image)
