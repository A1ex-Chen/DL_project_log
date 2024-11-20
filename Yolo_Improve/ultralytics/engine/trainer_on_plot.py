def on_plot(self, name, data=None):
    """Registers plots (e.g. to be consumed in callbacks)"""
    path = Path(name)
    self.plots[path] = {'data': data, 'timestamp': time.time()}
