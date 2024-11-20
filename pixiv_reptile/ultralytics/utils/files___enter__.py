def __enter__(self):
    """Changes the current directory to the specified directory."""
    os.chdir(self.dir)
