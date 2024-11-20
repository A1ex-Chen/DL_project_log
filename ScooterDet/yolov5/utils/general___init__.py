def __init__(self, new_dir):
    self.dir = new_dir
    self.cwd = Path.cwd().resolve()
