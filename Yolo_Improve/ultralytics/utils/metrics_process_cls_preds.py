def process_cls_preds(self, preds, targets):
    """
        Update confusion matrix for classification task.

        Args:
            preds (Array[N, min(nc,5)]): Predicted class labels.
            targets (Array[N, 1]): Ground truth class labels.
        """
    preds, targets = torch.cat(preds)[:, 0], torch.cat(targets)
    for p, t in zip(preds.cpu().numpy(), targets.cpu().numpy()):
        self.matrix[p][t] += 1
