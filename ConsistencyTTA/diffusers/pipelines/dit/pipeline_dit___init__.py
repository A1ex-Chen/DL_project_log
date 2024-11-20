def __init__(self, transformer: Transformer2DModel, vae: AutoencoderKL,
    scheduler: KarrasDiffusionSchedulers, id2label: Optional[Dict[int, str]
    ]=None):
    super().__init__()
    self.register_modules(transformer=transformer, vae=vae, scheduler=scheduler
        )
    self.labels = {}
    if id2label is not None:
        for key, value in id2label.items():
            for label in value.split(','):
                self.labels[label.lstrip().rstrip()] = int(key)
        self.labels = dict(sorted(self.labels.items()))
