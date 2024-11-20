def get_loss_acc(self, pred, gt, smoothing=True):
    gt = gt.contiguous().view(-1).long()
    if smoothing:
        eps = 0.2
        n_class = pred.size(1)
        one_hot = torch.zeros_like(pred).scatter(1, gt.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)
        loss = -(one_hot * log_prb).sum(dim=1).mean()
    else:
        loss = self.loss_ce(pred, gt.long())
    pred = pred.argmax(-1)
    acc = (pred == gt).sum() / float(gt.size(0))
    return loss, acc * 100
