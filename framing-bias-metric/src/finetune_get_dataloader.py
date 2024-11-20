def get_dataloader(self, type_path: str, batch_size: int, shuffle: bool=False
    ) ->DataLoader:
    dataset = self.get_dataset(type_path)
    if (self.hparams.sortish_sampler and type_path != 'test' and type_path !=
        'val'):
        sampler = dataset.make_sortish_sampler(batch_size, distributed=self
            .hparams.gpus > 1)
        return DataLoader(dataset, batch_size=batch_size, collate_fn=
            dataset.collate_fn, shuffle=False, num_workers=self.num_workers,
            sampler=sampler)
    elif self.hparams.max_tokens_per_batch is not None and type_path != 'test' and type_path != 'val':
        batch_sampler = dataset.make_dynamic_sampler(self.hparams.
            max_tokens_per_batch, distributed=self.hparams.gpus > 1)
        return DataLoader(dataset, batch_sampler=batch_sampler, collate_fn=
            dataset.collate_fn, num_workers=self.num_workers)
    else:
        return DataLoader(dataset, batch_size=batch_size, collate_fn=
            dataset.collate_fn, shuffle=shuffle, num_workers=self.
            num_workers, sampler=None)
