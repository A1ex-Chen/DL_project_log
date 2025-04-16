def __iter__(self):
    iteration = self.start_iter
    while iteration < self.num_iterations:
        if hasattr(self.batch_sampler.sampler, 'set_epoch'):
            self.batch_sampler.sampler.set_epoch(iteration)
        for batch in self.batch_sampler:
            yield batch
            iteration += 1
            if iteration >= self.num_iterations:
                break
