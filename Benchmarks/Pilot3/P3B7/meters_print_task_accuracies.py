def print_task_accuracies(self):
    for task, acc in self.accuracies.items():
        print(f'\t{task}: {acc}')
