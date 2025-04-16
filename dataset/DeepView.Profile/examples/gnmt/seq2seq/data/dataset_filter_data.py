def filter_data(self, min_len, max_len):
    """
        Preserves only samples which satisfy the following inequality:
            min_len <= src sample sequence length <= max_len AND
            min_len <= tgt sample sequence length <= max_len

        :param min_len: minimum sequence length
        :param max_len: maximum sequence length
        """
    logging.info(f'Filtering data, min len: {min_len}, max len: {max_len}')
    initial_len = len(self.src)
    filtered_src = []
    filtered_tgt = []
    for src, tgt in zip(self.src, self.tgt):
        if min_len <= len(src) <= max_len and min_len <= len(tgt) <= max_len:
            filtered_src.append(src)
            filtered_tgt.append(tgt)
    self.src = filtered_src
    self.tgt = filtered_tgt
    filtered_len = len(self.src)
    logging.info(f'Pairs before: {initial_len}, after: {filtered_len}')
