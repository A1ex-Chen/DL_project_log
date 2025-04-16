def __init__(self, sampler, batch_size, drop_last, rank=-1, world_size=2,
    wrap_last=False, gradient_accumulation_steps=None):
    super(DistributedBatchSampler, self).__init__(sampler, batch_size,
        drop_last)
    if rank == -1:
        assert False, 'should not be here'
    self.rank = rank
    self.world_size = world_size
    self.sampler.wrap_around = 0
    self.wrap_around = 0
    self.wrap_last = wrap_last
    self.start_iter = 0
    self.effective_batch_size = (batch_size if gradient_accumulation_steps is
        None else batch_size * gradient_accumulation_steps)
