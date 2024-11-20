def _get_train_dataset(self):
    dataset = AETrainDataset(self.train, item_count=self.item_count)
    return dataset
