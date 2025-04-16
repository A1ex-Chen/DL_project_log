def load(self):
    """Loads settings from the YAML file."""
    super().update(yaml_load(self.file))
