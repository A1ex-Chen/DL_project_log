def on_plot(self, name, data=None):
    """Registers plots (e.g. to be consumed in callbacks)"""
    self.plots[Path(name)] = {'data': data, 'timestamp': time.time()}
