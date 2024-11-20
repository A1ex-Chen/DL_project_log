def __init__(self, notes_encoder: SpectrogramNotesEncoder,
    continuous_encoder: SpectrogramContEncoder, decoder: T5FilmDecoder,
    scheduler: DDPMScheduler, melgan: (OnnxRuntimeModel if
    is_onnx_available() else Any)) ->None:
    super().__init__()
    self.min_value = math.log(1e-05)
    self.max_value = 4.0
    self.n_dims = 128
    self.register_modules(notes_encoder=notes_encoder, continuous_encoder=
        continuous_encoder, decoder=decoder, scheduler=scheduler, melgan=melgan
        )
