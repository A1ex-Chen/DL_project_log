def forward(self, labels, preds, weights=None):
    if self._encode_background_as_zeros:
        assert self._use_sigmoid_score is True
        total_scores = torch.sigmoid(preds)
    elif self._use_sigmoid_score:
        total_scores = torch.sigmoid(preds)[..., 1:]
    else:
        total_scores = F.softmax(preds, dim=-1)[..., 1:]
    """
        if preds.shape[self._dim] == 1:  # BCE
            scores = torch.sigmoid(preds)
        else:
            # assert preds.shape[
            #     self._dim] == 2, "precision only support 2 class"
            # TODO: add support for [N, C, ...] format.
            # TODO: add multiclass support
            if self._use_sigmoid_score:
                scores = torch.sigmoid(preds)[:, ..., 1:].sum(-1)
            else:
                scores = F.softmax(preds, dim=self._dim)[:, ..., 1:].sum(-1)
        """
    scores = torch.max(total_scores, dim=-1)[0]
    if weights is None:
        weights = (labels != self._ignore_idx).float()
    else:
        weights = weights.float()
    for i, thresh in enumerate(self._thresholds):
        tp, tn, fp, fn = _calc_binary_metrics(labels, scores, weights, self
            ._ignore_idx, thresh)
        rec_count = tp + fn
        prec_count = tp + fp
        if rec_count > 0:
            self.rec_count[i] += rec_count
            self.rec_total[i] += tp
        if prec_count > 0:
            self.prec_count[i] += prec_count
            self.prec_total[i] += tp
    return self.value
