def close(self):
    """Closes the progress bar."""
    if self.parent is None and self.output is not None:
        self.output.update(disp.HTML(''))
