def reset(self):
    self.confusion_matrix = torch.zeros(self.num_classes, self.num_classes,
        dtype=torch.int64, device='cpu')
    self._num_examples = 0
