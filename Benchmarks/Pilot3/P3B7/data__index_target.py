def _index_target(self, idx):
    return {task: target[idx] for task, target in self.target.items()}
