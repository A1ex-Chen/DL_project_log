def _get_eval_loader(self, mode):
    batch_size = (self.args.val_batch_size if mode == 'val' else self.args.
        test_batch_size)
    dataset = self._get_eval_dataset(mode)
    dataloader = data_utils.DataLoader(dataset, batch_size=batch_size,
        shuffle=False, pin_memory=True)
    return dataloader
