def __exit__(self, exc_type, value, traceback):
    """Defines behavior when exiting a 'with' block, prints error message if necessary."""
    if self.verbose and value:
        print(emojis(f"{self.msg}{': ' if self.msg else ''}{value}"))
    return True
