@property
def dtype(self):
    return self.visual.conv1.weight.dtype
