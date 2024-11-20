def torch_pad_sequence(sequence, padding_value, batch_first=True,
    padding_side='right'):
    if padding_side == 'right':
        sequence = torch.nn.utils.rnn.pad_sequence(sequence, batch_first=
            batch_first, padding_value=padding_value)
    elif padding_side == 'left':
        sequence = torch.nn.utils.rnn.pad_sequence([v.flip(-1) for v in
            sequence], batch_first=batch_first, padding_value=padding_value)
        sequence = sequence.flip(-1)
    else:
        raise NotImplementedError(f'padding_size={padding_side}')
    return sequence
