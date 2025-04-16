@staticmethod
def _concat(x, out):
    return torch.cat((x, out), 1)
