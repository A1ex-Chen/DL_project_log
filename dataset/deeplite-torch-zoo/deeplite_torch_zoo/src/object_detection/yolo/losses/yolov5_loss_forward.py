def forward(self, pred, true):
    loss = self.loss_fcn(pred, true)
    pred_prob = torch.sigmoid(pred)
    alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
    modulating_factor = torch.abs(true - pred_prob) ** self.gamma
    loss *= alpha_factor * modulating_factor
    if self.reduction == 'mean':
        return loss.mean()
    elif self.reduction == 'sum':
        return loss.sum()
    else:
        return loss
