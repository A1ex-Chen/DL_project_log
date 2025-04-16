@num_labels.setter
def num_labels(self, num_labels: int):
    self.id2label = {i: 'LABEL_{}'.format(i) for i in range(num_labels)}
    self.label2id = dict(zip(self.id2label.values(), self.id2label.keys()))
