def format(self, record):
    """Sets up logging with UTF-8 encoding and configurable verbosity."""
    return emojis(super().format(record))
