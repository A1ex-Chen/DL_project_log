def extract(*args, **kwargs):


    class Out:

        def __init__(self):
            self.pixel_values = torch.ones([0])

        def to(self, device):
            self.pixel_values.to(device)
            return self
    return Out()
