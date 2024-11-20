def update(self, accuracies, batch_size):
    for task, acc in accuracies.items():
        self.meters[task].update(acc[0].item(), batch_size)
