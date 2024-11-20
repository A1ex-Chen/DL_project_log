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
