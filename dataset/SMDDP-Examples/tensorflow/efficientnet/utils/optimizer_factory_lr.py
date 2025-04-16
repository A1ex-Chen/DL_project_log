@lr.setter
def lr(self, lr):
    self._optimizer._set_hyper('learning_rate', lr)
