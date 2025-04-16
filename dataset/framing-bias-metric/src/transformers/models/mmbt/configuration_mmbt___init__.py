def __init__(self, config, num_labels=None, modal_hidden_size=2048):
    self.__dict__ = config.__dict__
    self.modal_hidden_size = modal_hidden_size
    if num_labels:
        self.num_labels = num_labels
