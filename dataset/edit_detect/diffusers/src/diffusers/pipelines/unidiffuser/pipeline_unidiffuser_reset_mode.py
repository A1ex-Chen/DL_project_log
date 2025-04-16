def reset_mode(self):
    """Removes a manually set mode; after calling this, the pipeline will infer the mode from inputs."""
    self.mode = None
