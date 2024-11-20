def roll_out(self, x, n_steps, y_gt=None):
    y_pred = []
    for i in range(n_steps):
        y_pred_i = self.forward(x)
        y_pred.append(y_pred_i)
        if y_gt is not None:
            x = torch.cat([x[:, 1:, :], y_gt[:, i, :].unsqueeze(1)], dim=1)
        else:
            x = torch.cat([x[:, 1:, :], y_pred_i.unsqueeze(1)], dim=1)
    return torch.stack(y_pred, dim=1)
