def train_dataloader(self):
    loader = DataLoader(self.train_dataset, batch_size=self.batch_size,
        shuffle=True)
    return loader
