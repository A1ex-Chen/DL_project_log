def step(self):
    self.optimizer.step()
    self.scheduler.step()
