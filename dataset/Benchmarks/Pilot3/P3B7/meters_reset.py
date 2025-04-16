def reset(self):
    self.correct = {task: (0) for task, _ in self.tasks.items()}
    self.accuracies = {}
