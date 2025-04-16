def __next__(self):
    try:
        data = next(self.iter_loader)
    except StopIteration:
        self._epoch += 1
        if hasattr(self._dataloader.sampler, 'set_epoch'
            ) and self._use_distributed:
            self._dataloader.sampler.set_epoch(self._epoch)
        time.sleep(2)
        self.iter_loader = iter(self._dataloader)
        data = next(self.iter_loader)
    return data
