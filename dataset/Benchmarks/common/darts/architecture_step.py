def step(self, x_train, target_train, x_valid, target_valid, eta, optimizer,
    unrolled):
    """
        update alpha parameter by manually computing the gradients
        :param x_train:
        :param target_train:
        :param x_valid:
        :param target_valid:
        :param eta:
        :param optimizer: theta optimizer
        :param unrolled:
        :return:
        """
    self.optimizer.zero_grad()
    if unrolled:
        self.backward_step_unrolled(x_train, target_train, x_valid,
            target_valid, eta, optimizer)
    else:
        self.backward_step(x_valid, target_valid)
    self.optimizer.step()
