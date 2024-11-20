def __exit__(self, exc_type, exc_val, exc_tb):
    torch.nn.Module._slow_forward = self.original_slow_forward
