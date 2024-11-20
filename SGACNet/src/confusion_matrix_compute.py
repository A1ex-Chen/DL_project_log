def compute(self):
    if self._num_examples == 0:
        raise NotComputableError(
            'Confusion matrix must have at least one example before it can be computed.'
            )
    if self.average:
        self.confusion_matrix = self.confusion_matrix.float()
        if self.average == 'samples':
            return self.confusion_matrix / self._num_examples
        elif self.average == 'recall':
            return self.confusion_matrix / (self.confusion_matrix.sum(dim=1
                ) + 1e-15)
        elif self.average == 'precision':
            return self.confusion_matrix / (self.confusion_matrix.sum(dim=0
                ) + 1e-15)
    return self.confusion_matrix
