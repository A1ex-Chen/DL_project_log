@contextlib.contextmanager
def set_torch_seed(seed: int=42):
    saved_seed = torch.seed()
    torch.manual_seed(seed)
    yield
    torch.manual_seed(saved_seed)
