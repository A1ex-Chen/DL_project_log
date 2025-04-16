def test_dataloader(self):
    loader = DataLoader(self.test_dataset, batch_size=self.batch_size,
        shuffle=False)
    return loader
