def __init__(self, gpu0_bsz, *args, **kwargs):
    self.gpu0_bsz = gpu0_bsz
    super().__init__(*args, **kwargs)
