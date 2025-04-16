@learning_rate.setter
def learning_rate(self, learning_rate):
    self._optimizer._set_hyper('learning_rate', learning_rate)
