def set_scheduler(self, scheduler_type: str):
    library = importlib.import_module('k_diffusion')
    sampling = getattr(library, 'sampling')
    self.sampler = getattr(sampling, scheduler_type)
