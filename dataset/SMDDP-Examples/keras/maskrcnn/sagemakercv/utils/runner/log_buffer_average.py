def average(self, n=0):
    """Average latest n values or all values"""
    assert n >= 0
    for key in self.val_history:
        if 'time' in key:
            self.output[key] = self.val_history[key][-1]
        elif 'image' not in key:
            values = np.array(self.val_history[key][-n:])
            nums = np.array(self.n_history[key][-n:])
            avg = np.sum(values * nums) / np.sum(nums)
            self.output[key] = avg
    self.ready = True
