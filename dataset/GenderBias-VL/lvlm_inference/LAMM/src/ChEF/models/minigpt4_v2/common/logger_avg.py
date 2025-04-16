@property
def avg(self):
    d = torch.tensor(list(self.deque), dtype=torch.float32)
    return d.mean().item()
