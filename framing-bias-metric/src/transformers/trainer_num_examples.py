def num_examples(self, dataloader: DataLoader) ->int:
    """
        Helper to get number of samples in a :class:`~torch.utils.data.DataLoader` by accessing its dataset.

        Will raise an exception if the underlying dataset dese not implement method :obj:`__len__`
        """
    return len(dataloader.dataset)
