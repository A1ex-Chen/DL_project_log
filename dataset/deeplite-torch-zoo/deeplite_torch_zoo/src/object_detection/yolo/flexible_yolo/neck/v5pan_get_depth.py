def get_depth(self, n):
    return max(round(n * self.gd), 1) if n > 1 else n
