def fix_all_seeds(self, seed: int):
    """Fix all seeds to the same seed"""
    self.s.pythonhash = seed
    self.s.pythonrand = seed
    self.s.numpy = seed
    self.s.torch = seed
    self.set_seeds()
