def _make_sparse_tensor(self, query_logits, last_ys, last_xs, anchors,
    feature_value):
    if last_ys is None:
        N, _, qh, qw = query_logits.size()
        assert N == 1
        prob = torch.sigmoid_(query_logits).view(-1)
        pidxs = torch.where(prob > self.score_th)[0]
        y = torch.div(pidxs, qw).int()
        x = torch.remainder(pidxs, qw).int()
    else:
        prob = torch.sigmoid_(query_logits).view(-1)
        pidxs = prob > self.score_th
        y = last_ys[pidxs]
        x = last_xs[pidxs]
    if y.size(0) == 0:
        return None, None, None, None, None, None
    _, fc, fh, fw = feature_value.shape
    ys, xs = [], []
    for i in range(2):
        for j in range(2):
            ys.append(y * 2 + i)
            xs.append(x * 2 + j)
    ys = torch.cat(ys, dim=0)
    xs = torch.cat(xs, dim=0)
    inds = (ys * fw + xs).long()
    sparse_ys = []
    sparse_xs = []
    for i in range(-1 * self.context, self.context + 1):
        for j in range(-1 * self.context, self.context + 1):
            sparse_ys.append(ys + i)
            sparse_xs.append(xs + j)
    sparse_ys = torch.cat(sparse_ys, dim=0)
    sparse_xs = torch.cat(sparse_xs, dim=0)
    good_idx = (sparse_ys >= 0) & (sparse_ys < fh) & (sparse_xs >= 0) & (
        sparse_xs < fw)
    sparse_ys = sparse_ys[good_idx]
    sparse_xs = sparse_xs[good_idx]
    sparse_yx = torch.stack((sparse_ys, sparse_xs), dim=0).t()
    sparse_yx = torch.unique(sparse_yx, sorted=False, dim=0)
    sparse_ys = sparse_yx[:, 0]
    sparse_xs = sparse_yx[:, 1]
    sparse_inds = (sparse_ys * fw + sparse_xs).long()
    sparse_features = feature_value.view(fc, -1).transpose(0, 1)[sparse_inds
        ].view(-1, fc)
    sparse_indices = torch.stack((torch.zeros_like(sparse_ys), sparse_ys,
        sparse_xs), dim=-1)
    sparse_tensor = spconv.SparseConvTensor(sparse_features, sparse_indices
        .int(), (fh, fw), 1)
    anchors = anchors.tensor.view(-1, self.anchor_num, 4)
    selected_anchors = anchors[inds].view(1, -1, 4)
    return sparse_tensor, ys, xs, inds, selected_anchors, sparse_indices.size(0
        )
