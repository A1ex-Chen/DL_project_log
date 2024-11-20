def make_sortish_sampler(self, batch_size, distributed=False, shuffle=True,
    **kwargs):
    if distributed:
        return DistributedSortishSampler(self, batch_size, shuffle=shuffle,
            **kwargs)
    else:
        return SortishSampler(self.src_lens, batch_size, shuffle=shuffle)
