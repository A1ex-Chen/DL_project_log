@property
def median(self):
    d = torch.tensor(list(self.deque))
    return d.median().item()
