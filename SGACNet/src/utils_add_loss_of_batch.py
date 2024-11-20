def add_loss_of_batch(self, inputs, targets):
    targets_m = targets.clone()
    targets_m -= 1
    loss = self.ce_loss(inputs, targets_m.long())
    self.total_loss += loss
    self.nr_pixels += torch.sum(targets_m >= 0)
