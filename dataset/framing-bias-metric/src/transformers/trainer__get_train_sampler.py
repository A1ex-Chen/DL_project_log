def _get_train_sampler(self) ->Optional[torch.utils.data.sampler.Sampler]:
    if isinstance(self.train_dataset, torch.utils.data.IterableDataset
        ) or not isinstance(self.train_dataset, collections.abc.Sized):
        return None
    elif is_torch_tpu_available():
        return get_tpu_sampler(self.train_dataset)
    else:
        return RandomSampler(self.train_dataset
            ) if self.args.local_rank == -1 else DistributedSampler(self.
            train_dataset)
