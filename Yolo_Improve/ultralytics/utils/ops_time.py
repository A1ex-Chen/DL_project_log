def time(self):
    """Get current time."""
    if self.cuda:
        torch.cuda.synchronize(self.device)
    return time.time()
