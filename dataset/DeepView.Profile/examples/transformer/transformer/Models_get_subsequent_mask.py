def get_subsequent_mask(seq):
    """ For masking out the subsequent info. """
    sz_b, len_s = seq.size()
    subsequent_mask = torch.triu(torch.ones((len_s, len_s), device=seq.
        device, dtype=torch.uint8), diagonal=1)
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1)
    return subsequent_mask
