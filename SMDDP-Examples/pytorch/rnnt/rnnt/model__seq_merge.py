def _seq_merge(self, x):
    assert len(x) == 2, 'Only two segment seq split is supprorted now'
    x1_pad = torch.nn.functional.pad(x[1], (0, 0, 0, x[0].size(1) - x[1].
        size(1)))
    y = torch.cat((x[0], x1_pad), dim=0)
    return y
