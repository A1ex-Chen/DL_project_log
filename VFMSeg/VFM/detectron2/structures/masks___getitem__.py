def __getitem__(self, item) ->'ROIMasks':
    """
        Returns:
            ROIMasks: Create a new :class:`ROIMasks` by indexing.

        The following usage are allowed:

        1. `new_masks = masks[2:10]`: return a slice of masks.
        2. `new_masks = masks[vector]`, where vector is a torch.BoolTensor
           with `length = len(masks)`. Nonzero elements in the vector will be selected.

        Note that the returned object might share storage with this object,
        subject to Pytorch's indexing semantics.
        """
    t = self.tensor[item]
    if t.dim() != 3:
        raise ValueError(
            f'Indexing on ROIMasks with {item} returns a tensor with shape {t.shape}!'
            )
    return ROIMasks(t)
