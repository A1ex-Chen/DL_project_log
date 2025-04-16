@torch.jit.script
def foo(x, y):
    return torch.sigmoid(x) + y
