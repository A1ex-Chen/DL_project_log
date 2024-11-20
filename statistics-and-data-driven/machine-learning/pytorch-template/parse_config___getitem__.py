def __getitem__(self, name):
    """Access items like ordinary dict."""
    return self.config[name]
