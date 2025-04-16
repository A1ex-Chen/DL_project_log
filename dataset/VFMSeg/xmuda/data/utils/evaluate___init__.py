def __init__(self, class_names, labels=None):
    self.class_names = tuple(class_names)
    self.num_classes = len(class_names)
    self.labels = np.arange(self.num_classes) if labels is None else np.array(
        labels)
    assert self.labels.shape[0] == self.num_classes
    self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))
