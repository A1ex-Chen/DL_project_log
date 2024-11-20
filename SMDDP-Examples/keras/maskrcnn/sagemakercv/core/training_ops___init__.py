def __init__(self, batch_size_per_im=512, fg_fraction=0.25, fg_thresh=0.5,
    bg_thresh_hi=0.5, bg_thresh_lo=0.0):
    self.batch_size_per_im = batch_size_per_im
    self.fg_fraction = fg_fraction
    self.fg_thresh = fg_thresh
    self.bg_thresh_hi = bg_thresh_hi
    self.bg_thresh_lo = bg_thresh_lo
