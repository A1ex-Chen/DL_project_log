def save(self):
    """Saves the current settings to the YAML file."""
    yaml_save(self.file, dict(self))
