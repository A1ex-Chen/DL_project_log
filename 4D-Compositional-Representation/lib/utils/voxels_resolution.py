@property
def resolution(self):
    assert self.data.shape[0] == self.data.shape[1] == self.data.shape[2]
    return self.data.shape[0]
