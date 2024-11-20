def update(self, y, y_pred, num_examples=1):
    assert len(y) == len(y_pred
        ), 'label and prediction need to have the same size'
    self._num_examples += num_examples
    y = y.type(self.dtype)
    y_pred = y_pred.type(self.dtype)
    indices = self.num_classes * y + y_pred
    m = torch.bincount(indices, minlength=self.num_classes ** 2)
    m = m.reshape(self.num_classes, self.num_classes)
    self.confusion_matrix += m.to(self.confusion_matrix)
