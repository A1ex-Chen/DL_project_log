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
