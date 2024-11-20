def __init__(self, num_classes, average=None, output_transform=lambda x: x):
    if average is not None and average not in ('samples', 'recall', 'precision'
        ):
        raise ValueError(
            "Argument average can None or one of ['samples', 'recall', 'precision']"
            )
    self.num_classes = num_classes
    if self.num_classes < np.sqrt(2 ** 8):
        self.dtype = torch.uint8
    elif self.num_classes < np.sqrt(2 ** 16 / 2):
        self.dtype = torch.int16
    elif self.num_classes < np.sqrt(2 ** 32 / 2):
        self.dtype = torch.int32
    else:
        self.dtype = torch.int64
    self._num_examples = 0
    self.average = average
    self.confusion_matrix = None
    super(ConfusionMatrixPytorch, self).__init__(output_transform=
        output_transform)
