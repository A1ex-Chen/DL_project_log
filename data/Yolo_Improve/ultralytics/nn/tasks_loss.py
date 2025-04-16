def loss(self, batch, preds=None):
    """
        Compute loss.

        Args:
            batch (dict): Batch to compute loss on.
            preds (torch.Tensor | List[torch.Tensor]): Predictions.
        """
    if not hasattr(self, 'criterion'):
        self.criterion = self.init_criterion()
    if preds is None:
        preds = self.forward(batch['img'], txt_feats=batch['txt_feats'])
    return self.criterion(preds, batch)
