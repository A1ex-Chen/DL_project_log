def __init__(self, model):
    super().__init__()
    self.model = model
    self.num_labels = 2
    self.qa_outputs = torch.nn.Linear(self.model.config.hidden_size, self.
        num_labels)
