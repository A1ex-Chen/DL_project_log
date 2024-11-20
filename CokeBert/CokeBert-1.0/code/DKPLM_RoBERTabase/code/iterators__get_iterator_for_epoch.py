def _get_iterator_for_epoch(self, epoch, shuffle, fix_batches_to_gpus=False):

    def shuffle_batches(batches, seed):
        with data_utils.numpy_seed(seed):
            np.random.shuffle(batches)
        return batches
    if self._supports_prefetch:
        batches = self.frozen_batches
        if shuffle and not fix_batches_to_gpus:
            batches = shuffle_batches(list(batches), self.seed + epoch)
        batches = list(ShardedIterator(batches, self.num_shards, self.
            shard_id, fill_value=[]))
        self.dataset.prefetch([i for s in batches for i in s])
        if shuffle and fix_batches_to_gpus:
            batches = shuffle_batches(batches, self.seed + epoch + self.
                shard_id)
    else:
        if shuffle:
            batches = shuffle_batches(list(self.frozen_batches), self.seed +
                epoch)
        else:
            batches = self.frozen_batches
        batches = ShardedIterator(batches, self.num_shards, self.shard_id,
            fill_value=[])
    return CountingIterator(torch.utils.data.DataLoader(self.dataset,
        collate_fn=self.collate_fn, batch_sampler=batches))
