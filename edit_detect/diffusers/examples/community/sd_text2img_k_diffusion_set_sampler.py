def set_sampler(self, scheduler_type: str):
    warnings.warn(
        'The `set_sampler` method is deprecated, please use `set_scheduler` instead.'
        )
    return self.set_scheduler(scheduler_type)
