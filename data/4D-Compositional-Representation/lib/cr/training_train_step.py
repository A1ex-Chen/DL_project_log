def train_step(self, data):
    """ Performs a train step.

        Args:
            data (tensor): training data
        """
    self.model.train()
    self.optimizer.zero_grad()
    loss = self.compute_loss(data)
    loss.backward()
    self.optimizer.step()
    return loss.item()
