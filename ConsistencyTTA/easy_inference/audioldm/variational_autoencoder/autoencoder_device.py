@property
def device(self):
    return next(self.parameters()).device
