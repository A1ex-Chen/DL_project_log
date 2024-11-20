def ngrams(sequence, n, pad_left=False, pad_right=False, left_pad_symbol=
    None, right_pad_symbol=None):
    sequence = pad_sequence(sequence, n, pad_left, pad_right,
        left_pad_symbol, right_pad_symbol)
    history = []
    while n > 1:
        history.append(next(sequence))
        n -= 1
    for item in sequence:
        history.append(item)
        yield tuple(history)
        del history[0]
