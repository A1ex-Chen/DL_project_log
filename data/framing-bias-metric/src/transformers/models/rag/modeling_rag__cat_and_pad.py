@staticmethod
def _cat_and_pad(tensors, pad_token_id):
    output = tensors[0].new(sum([t.shape[0] for t in tensors]), max([t.
        shape[1] for t in tensors])).fill_(pad_token_id)
    ind = 0
    for t in tensors:
        output[ind:ind + t.shape[0], :t.shape[1]] = t
        ind += t.shape[0]
    return output
