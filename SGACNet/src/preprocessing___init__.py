def __init__(self, downsampling_rates=None):
    if downsampling_rates is None:
        self.downsampling_rates = [8, 16, 32]
    else:
        self.downsampling_rates = downsampling_rates
