def _get_eval_dataset(self, mode):
    answers = self.val if mode == 'val' else self.test
    dataset = BertEvalDataset(self.train, answers, self.max_len, self.
        CLOZE_MASK_TOKEN, self.test_negative_samples)
    return dataset
