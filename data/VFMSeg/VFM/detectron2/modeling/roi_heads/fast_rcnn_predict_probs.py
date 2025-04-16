def predict_probs(self, predictions: Tuple[torch.Tensor, torch.Tensor],
    proposals: List[Instances]):
    """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were
                used to compute predictions.

        Returns:
            list[Tensor]:
                A list of Tensors of predicted class probabilities for each image.
                Element i has shape (Ri, K + 1), where Ri is the number of proposals for image i.
        """
    scores, _ = predictions
    num_inst_per_image = [len(p) for p in proposals]
    probs = F.softmax(scores, dim=-1)
    return probs.split(num_inst_per_image, dim=0)
