def _get_train_loader(self):
    dataset = self._get_train_dataset()
    dataloader = data_utils.DataLoader(dataset, batch_size=self.args.
        train_batch_size, shuffle=True, pin_memory=True)
    return dataloader
