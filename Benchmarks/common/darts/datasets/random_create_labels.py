def create_labels(self, tasks, num_samples):
    labels = {}
    for task, num_classes in tasks.items():
        labels[task] = np.random.randint(num_classes, size=num_samples)
    return labels
