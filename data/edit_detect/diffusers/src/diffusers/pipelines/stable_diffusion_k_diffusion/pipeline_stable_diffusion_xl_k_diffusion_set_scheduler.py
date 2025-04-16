def set_scheduler(self, scheduler_type: str):
    library = importlib.import_module('k_diffusion')
    sampling = getattr(library, 'sampling')
    try:
        self.sampler = getattr(sampling, scheduler_type)
    except Exception:
        valid_samplers = []
        for s in dir(sampling):
            if 'sample_' in s:
                valid_samplers.append(s)
        raise ValueError(
            f'Invalid scheduler type {scheduler_type}. Please choose one of {valid_samplers}.'
            )
