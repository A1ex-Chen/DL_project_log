def __init__(self, root_dir, scenes):
    self.root_dir = root_dir
    self.data = []
    self.glob_frames(scenes)
