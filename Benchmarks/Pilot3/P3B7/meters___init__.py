def __init__(self, tasks, dataloader):
    self.tasks = tasks
    self.loader = dataloader
    self.reset()
