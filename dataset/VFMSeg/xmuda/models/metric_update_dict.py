def update_dict(self, preds, labels):
    seg_logit = preds['seg_logit']
    seg_label = labels['seg_label']
    pred_label = seg_logit.argmax(1)
    mask = seg_label != self.ignore_index
    seg_label = seg_label[mask]
    pred_label = pred_label[mask]
    n = self.num_classes
    with torch.no_grad():
        if self.mat is None:
            self.mat = seg_label.new_zeros((n, n))
        inds = n * seg_label + pred_label
        self.mat += torch.bincount(inds, minlength=n ** 2).reshape(n, n)
