def update_accuracy(self):
    for task, correct in self.correct.items():
        acc = 100.0 * correct / len(self.loader.dataset)
        self.accuracies[task] = acc
