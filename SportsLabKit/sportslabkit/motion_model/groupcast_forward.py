def forward(self, x):
    assert x.dim() == 3 or x.dim() == 2
    if x.dim() == 2:
        x = x.unsqueeze(0)
    if x.shape[1] == 1:
        return x
    v = x[:, -self.obs_steps:].diff(dim=1).mean(dim=1)
    y_pred = x[:, -1] + v
    return y_pred
