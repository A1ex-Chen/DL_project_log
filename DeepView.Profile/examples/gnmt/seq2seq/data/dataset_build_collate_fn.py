def build_collate_fn(batch_first=False, parallel=True, sort=False):
    """
    Factory for collate_fn functions.

    :param batch_first: if True returns batches in (batch, seq) format, if
        False returns in (seq, batch) format
    :param parallel: if True builds batches from parallel corpus (src, tgt)
    :param sort: if True sorts by src sequence length within each batch
    """

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

    def parallel_collate(seqs):
        """
        Builds batches from parallel dataset (src, tgt), optionally sorts batch
        by src sequence length.

        :param seqs: tuple of (src, tgt) sequences
        """
        src_seqs, tgt_seqs = zip(*seqs)
        if sort:
            indices, src_seqs = zip(*sorted(enumerate(src_seqs), key=lambda
                item: len(item[1]), reverse=True))
            tgt_seqs = [tgt_seqs[idx] for idx in indices]
        return tuple([collate_seq(s) for s in [src_seqs, tgt_seqs]])

    def single_collate(src_seqs):
        """
        Builds batches from text dataset, optionally sorts batch by src
        sequence length.

        :param src_seqs: source sequences
        """
        if sort:
            indices, src_seqs = zip(*sorted(enumerate(src_seqs), key=lambda
                item: len(item[1]), reverse=True))
        else:
            indices = range(len(src_seqs))
        return collate_seq(src_seqs), tuple(indices)
    if parallel:
        return parallel_collate
    else:
        return single_collate
