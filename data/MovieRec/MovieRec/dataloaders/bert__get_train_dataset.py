def _get_train_dataset(self):
    dataset = BertTrainDataset(self.train, self.max_len, self.mask_prob,
        self.CLOZE_MASK_TOKEN, self.item_count, self.rng)
    return dataset
