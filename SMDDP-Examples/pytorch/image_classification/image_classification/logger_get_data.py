def get_data(self):
    if self.n == 0:
        return None, 0
    return self.val / self.n, self.n
