def _get_eval_dataset(self, mode):
    data = self.val if mode == 'val' else self.test
    dataset = AEEvalDataset(data, item_count=self.item_count)
    return dataset
