def __exit__(self, exc_type, exc_val, exc_tb):
    """Restore the current working directory on context exit."""
    os.chdir(self.cwd)
