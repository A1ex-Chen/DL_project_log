def __init__(self, save_dir=Path('.'), plot=False, on_plot=None, names=()
    ) ->None:
    """Initialize an OBBMetrics instance with directory, plotting, callback, and class names."""
    self.save_dir = save_dir
    self.plot = plot
    self.on_plot = on_plot
    self.names = names
    self.box = Metric()
    self.speed = {'preprocess': 0.0, 'inference': 0.0, 'loss': 0.0,
        'postprocess': 0.0}
