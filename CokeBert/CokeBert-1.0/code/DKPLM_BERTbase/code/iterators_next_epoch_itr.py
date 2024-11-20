def next_epoch_itr(self, shuffle=True, fix_batches_to_gpus=False):
    """Return a new iterator over the dataset.

        Args:
            shuffle (bool, optional): shuffle batches before returning the
                iterator. Default: ``True``
            fix_batches_to_gpus: ensure that batches are always
                allocated to the same shards across epochs. Requires
                that :attr:`dataset` supports prefetching. Default:
                ``False``
        """
    if self._next_epoch_itr is not None:
        self._cur_epoch_itr = self._next_epoch_itr
        self._next_epoch_itr = None
    else:
        self.epoch += 1
        self._cur_epoch_itr = self._get_iterator_for_epoch(self.epoch,
            shuffle, fix_batches_to_gpus=fix_batches_to_gpus)
    return self._cur_epoch_itr
