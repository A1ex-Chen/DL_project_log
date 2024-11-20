def time(self):
    """
        Get current time.
        """
    if self.cuda:
        torch.cuda.synchronize()
    return time.time()
