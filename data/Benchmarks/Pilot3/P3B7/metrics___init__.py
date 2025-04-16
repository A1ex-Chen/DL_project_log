def __init__(self, tasks, average='micro'):
    self.metrics = self._create_metrics(tasks, average)
