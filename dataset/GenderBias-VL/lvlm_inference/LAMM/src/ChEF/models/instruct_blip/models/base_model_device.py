@property
def device(self):
    return list(self.parameters())[0].device
