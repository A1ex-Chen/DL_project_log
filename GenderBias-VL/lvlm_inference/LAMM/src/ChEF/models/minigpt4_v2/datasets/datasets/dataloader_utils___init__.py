def __init__(self, dataloader: DataLoader, use_distributed: bool=False):
    self._dataloader = dataloader
    self.iter_loader = iter(self._dataloader)
    self._use_distributed = use_distributed
    self._epoch = 0
