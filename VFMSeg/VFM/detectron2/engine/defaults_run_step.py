def run_step(self):
    self._trainer.iter = self.iter
    self._trainer.run_step()
