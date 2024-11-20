def reset_epoch(self):
    v, n = self.epoch_aggregator.get_data()
    self.epoch_aggregator.reset()
    if v is not None:
        self.run_aggregator.record(v, n=n)
