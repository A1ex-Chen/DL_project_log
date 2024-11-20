def get_pytorch_dataloaders(self):
    train_loader = self._get_train_loader()
    val_loader = self._get_val_loader()
    test_loader = self._get_test_loader()
    return train_loader, val_loader, test_loader
