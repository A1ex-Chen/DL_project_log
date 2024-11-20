def _close_dataloader_mosaic(self):
    """Update dataloaders to stop using mosaic augmentation."""
    if hasattr(self.train_loader.dataset, 'mosaic'):
        self.train_loader.dataset.mosaic = False
    if hasattr(self.train_loader.dataset, 'close_mosaic'):
        LOGGER.info('Closing dataloader mosaic')
        self.train_loader.dataset.close_mosaic(hyp=self.args)
