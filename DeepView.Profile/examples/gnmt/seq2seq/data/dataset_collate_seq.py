def collate_seq(seq):
    """
        Builds batches for training or inference.
        Batches are returned as pytorch tensors, with padding.

        :param seq: list of sequences
        """
    lengths = [len(s) for s in seq]
    batch_length = max(lengths)
    shape = batch_length, len(seq)
    seq_tensor = torch.full(shape, config.PAD, dtype=torch.int64)
    for i, s in enumerate(seq):
        end_seq = lengths[i]
        seq_tensor[:end_seq, i].copy_(s[:end_seq])
    if batch_first:
        seq_tensor = seq_tensor.t()
    return seq_tensor, lengths
