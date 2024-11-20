def epoch_generator_wrapper(self, gen):
    for g in gen:
        self.start_epoch()
        yield g
        self.end_epoch()
