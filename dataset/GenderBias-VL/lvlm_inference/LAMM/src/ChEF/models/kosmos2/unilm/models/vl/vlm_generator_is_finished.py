def is_finished(self, step: int, unfin_idx: int, max_len: int,
    finalized_sent_len: int, beam_size: int):
    """
        Check whether decoding for a sentence is finished, which
        occurs when the list of finalized sentences has reached the
        beam size, or when we reach the maximum length.
        """
    assert finalized_sent_len <= beam_size
    if finalized_sent_len == beam_size or step == max_len:
        return True
    return False
