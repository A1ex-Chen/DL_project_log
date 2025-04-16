def prepare_extra_step_kwargs(self, eta, generator=None):
    accepts_eta = 'eta' in set(inspect.signature(self.scheduler.step).
        parameters.keys())
    extra_step_kwargs = {}
    if accepts_eta:
        extra_step_kwargs['eta'] = eta
    accepts_generator = 'generator' in set(inspect.signature(self.scheduler
        .step).parameters.keys())
    if accepts_generator:
        extra_step_kwargs['generator'] = generator
    return extra_step_kwargs
