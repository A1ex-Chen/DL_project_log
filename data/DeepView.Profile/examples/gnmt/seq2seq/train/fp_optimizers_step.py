def step(self, optimizer, scheduler, update=True):
    """
        Performs one step of the optimizer.

        :param loss: value of loss function
        :param optimizer: optimizer
        :param update: if True executes weight update
        """
    if update:
        if self.grad_clip != float('inf'):
            clip_grad_norm_(self.model.parameters(), self.grad_clip)
        optimizer.step()
        scheduler.step()
        self.model.zero_grad()
