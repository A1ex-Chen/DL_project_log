def _get_train_sampler(self) ->Optional[torch.utils.data.sampler.Sampler]:
    if isinstance(self.train_dataset, torch.utils.data.IterableDataset):
        return None
    elif is_torch_tpu_available():
        return get_tpu_sampler(self.train_dataset)
    else:
        if self.args.sortish_sampler:
            self.train_dataset.make_sortish_sampler(self.args.
                per_device_train_batch_size, distributed=self.args.
                parallel_mode == ParallelMode.DISTRIBUTED)
        return RandomSampler(self.train_dataset
            ) if self.args.local_rank == -1 else DistributedSampler(self.
            train_dataset)
