def reset_attnmaps(self):
    self.step = 0
    self.attnmaps = {}
    self.attnmaps_sizes = []
    self.attnmasks = {}
    self.maskready = False
    self.history = {}
