@property
def interrupted(self):
    if not dist.is_initialized():
        return self._interrupted
    interrupted = torch.tensor(self._interrupted).int().to(self.device)
    dist.broadcast(interrupted, 0)
    interrupted = bool(interrupted.item())
    return interrupted
