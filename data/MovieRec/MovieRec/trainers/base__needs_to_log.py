def _needs_to_log(self, accum_iter):
    return (accum_iter % self.log_period_as_iter < self.args.
        train_batch_size and accum_iter != 0)
