def __init__(self, x_dim: int, num_samples: int, tasks: Dict[str, int],
    seed: int=13):
    np.random.seed(seed)
    self.data = self.create_data(x_dim, num_samples)
    self.labels = self.create_labels(tasks, num_samples)
