def get_dataset(self, type_path) ->Seq2SeqDataset:
    n_obs = self.n_obs[type_path]
    max_target_length = self.target_lens[type_path]
    dataset = self.dataset_class(self.tokenizer, type_path=type_path, n_obs
        =n_obs, max_target_length=max_target_length, extra_task=self.
        extra_task, **self.dataset_kwargs)
    return dataset
