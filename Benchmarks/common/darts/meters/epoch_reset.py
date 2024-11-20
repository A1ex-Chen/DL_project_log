def reset(self):
    self.loss = []
    self.acc = {task: [] for task, _ in self.acc_meter.meters.items()}
