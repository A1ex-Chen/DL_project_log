def forward(self, pred, target):
    loss = self.loss_func(pred, target)
    return loss
