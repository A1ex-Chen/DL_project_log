def reset_iteration(self):
    v, n = self.iteration_aggregator.get_data()
    self.iteration_aggregator.reset()
    if v is not None:
        self.epoch_aggregator.record(v, n=n)
