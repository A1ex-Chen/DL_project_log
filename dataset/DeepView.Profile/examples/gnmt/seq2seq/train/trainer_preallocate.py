def preallocate(self, data_loader, training):
    """
        Generates maximum sequence length batch and runs forward and backward
        pass without updating model parameters.

        :param data_loader: data loader
        :param training: if True preallocates memory for backward pass
        """
    batch_size = data_loader.batch_size
    max_len = data_loader.dataset.max_len
    src_length = [max_len] * batch_size
    tgt_length = [max_len] * batch_size
    if self.batch_first:
        shape = batch_size, max_len
    else:
        shape = max_len, batch_size
    src = torch.full(shape, 4, dtype=torch.int64)
    tgt = torch.full(shape, 4, dtype=torch.int64)
    src = src, src_length
    tgt = tgt, tgt_length
    self.iterate(src, tgt, update=False, training=training)
    self.model.zero_grad()
