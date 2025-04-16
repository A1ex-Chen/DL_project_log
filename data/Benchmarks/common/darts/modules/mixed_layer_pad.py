def pad(self, tensors):
    """Pad with zeros for mixed layers"""
    prev = tensors[0]
    padded = []
    for tensor in tensors:
        if tensor.shape < prev.shape:
            tensor_pad = F.pad(input=tensor, pad=(1, 1, 1, 1), mode=
                'constant', value=0)
            padded.append(tensor_pad)
        else:
            padded.append(tensor)
        prev = tensor
    return padded
