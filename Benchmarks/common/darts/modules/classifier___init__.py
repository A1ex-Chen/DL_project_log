def __init__(self, input_dim: int, tasks: Dict[str, int]):
    super(MultitaskClassifier, self).__init__()
    self.tasks = tasks
    for task, num_classes in tasks.items():
        self.add_module(task, nn.Linear(input_dim, num_classes))
