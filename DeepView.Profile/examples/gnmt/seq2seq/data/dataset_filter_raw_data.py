def filter_raw_data(self, min_len, max_len):
    """
        Preserves only samples which satisfy the following inequality:
            min_len <= src sample sequence length <= max_len AND
            min_len <= tgt sample sequence length <= max_len

        :param min_len: minimum sequence length
        :param max_len: maximum sequence length
        """
    initial_len = len(self.raw_src)
    filtered_src = []
    filtered_tgt = []
    filtered_src_len = []
    filtered_tgt_len = []
    for src, tgt in zip(self.raw_src, self.raw_tgt):
        src_len = src.count(' ') + 1
        tgt_len = tgt.count(' ') + 1
        if min_len <= src_len <= max_len and min_len <= tgt_len <= max_len:
            filtered_src.append(src)
            filtered_tgt.append(tgt)
            filtered_src_len.append(src_len)
            filtered_tgt_len.append(tgt_len)
    self.raw_src = filtered_src
    self.raw_tgt = filtered_tgt
    self.src_len = filtered_src_len
    self.tgt_len = filtered_tgt_len
    filtered_len = len(self.raw_src)
    logging.info(f'Pairs before: {initial_len}, after: {filtered_len}')
