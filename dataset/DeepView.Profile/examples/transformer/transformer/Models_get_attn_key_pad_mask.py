def get_attn_key_pad_mask(seq_k, seq_q):
    """ For masking out the padding part of key sequence. """
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(Constants.PAD)
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)
    return padding_mask
