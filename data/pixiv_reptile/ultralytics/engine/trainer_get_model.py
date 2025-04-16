def get_model(self, cfg=None, weights=None, verbose=True):
    """Get model and raise NotImplementedError for loading cfg files."""
    raise NotImplementedError(
        "This task trainer doesn't support loading cfg files")
