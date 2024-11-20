@register_to_config
def __init__(self, x_res: int=256, y_res: int=256, sample_rate: int=22050,
    n_fft: int=2048, hop_length: int=512, top_db: int=80, n_iter: int=32):
    self.hop_length = hop_length
    self.sr = sample_rate
    self.n_fft = n_fft
    self.top_db = top_db
    self.n_iter = n_iter
    self.set_resolution(x_res, y_res)
    self.audio = None
    if not _librosa_can_be_imported:
        raise ValueError(_import_error)
