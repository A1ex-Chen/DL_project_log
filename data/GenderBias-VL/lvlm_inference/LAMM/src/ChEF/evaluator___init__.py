def __init__(self, dataset, indices):
    super().__init__(dataset, indices)
    self.task_name = dataset.task_name
    self.dataset_name = dataset.dataset_name
    self.data = dataset.data
    if hasattr(dataset, 'system_msg'):
        self.system_msg = dataset.system_msg
    if hasattr(dataset, 'circularidx'):
        self.circularidx = dataset.circularidx
