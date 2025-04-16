def forward(self, data_batch, mixed_batch=None):
    if mixed_batch is None:
        feats = self.net_3d(data_batch['x'])
    else:
        feats = self.net_3d(mixed_batch)
    x = self.linear(feats)
    preds = {'feats': feats, 'seg_logit': x}
    if self.dual_head:
        preds['seg_logit2'] = self.linear2(feats)
    return preds
