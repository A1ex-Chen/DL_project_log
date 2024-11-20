def upsample(x, stride, target_len, separate_cls=True, truncate_seq=False):
    """
    Upsample tensor `x` to match `target_len` by repeating the tokens `stride` time on the sequence length dimension.
    """
    if stride == 1:
        return x
    if separate_cls:
        cls = x[:, :1]
        x = x[:, 1:]
    output = tf.repeat(x, repeats=stride, axis=1)
    if separate_cls:
        if truncate_seq:
            output = tf.pad(output, [[0, 0], [0, stride - 1], [0, 0]])
        output = output[:, :target_len - 1]
        output = tf.concat([cls, output], axis=1)
    else:
        output = output[:, :target_len]
    return output