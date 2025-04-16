def iteration_generator_wrapper(self, gen, mode='train'):
    for g in gen:
        self.start_iteration(mode=mode)
        yield g
        self.end_iteration(mode=mode)
