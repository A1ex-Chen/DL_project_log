def __init__(self, new_dir):
    """Sets the working directory to 'new_dir' upon instantiation."""
    self.dir = new_dir
    self.cwd = Path.cwd().resolve()
