def __iter__(self):
    return self.mixup_loader(self.dataloader)
