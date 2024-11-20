def __init__(self, cfg, torch_model):
    assert isinstance(torch_model, meta_arch.RetinaNet)
    super().__init__(cfg, torch_model)
