@property
def device(self):
    return list(self.parameters())[-1].device
