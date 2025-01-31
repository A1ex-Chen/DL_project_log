def dimshuffle(input_img, from_axes, to_axes):
    if from_axes.find('0') == -1 or from_axes.find('1') == -1:
        raise ValueError(
            "`from_axes` must contain both axis0 ('0') andaxis 1 ('1')")
    if to_axes.find('0') == -1 or to_axes.find('1') == -1:
        raise ValueError(
            "`to_axes` must contain both axis0 ('0') andaxis 1 ('1')")
    if len(from_axes) != len(input_img.shape):
        raise ValueError(
            'Number of axis given by `from_axes` does not match the number of axis in `input_img`'
            )
    to_axes_c = to_axes.find('c')
    from_axes_c = from_axes.find('c')
    if to_axes_c == -1 and from_axes_c >= 0:
        if input_img.shape[from_axes_c] != 1:
            raise ValueError(
                'Cannot remove channel axis because size is not equal to 1')
        input_img = input_img.squeeze(axis=from_axes_c)
        from_axes = from_axes.replace('c', '')
    to_axes_b = to_axes.find('b')
    from_axes_b = from_axes.find('b')
    if to_axes_b == -1 and from_axes_b >= 0:
        if input_img.shape[from_axes_b] != 1:
            raise ValueError(
                'Cannot remove batch axis because size is not equal to 1')
        input_img = input_img.squeeze(axis=from_axes_b)
        from_axes = from_axes.replace('b', '')
    if to_axes_b >= 0 and from_axes_b == -1:
        input_img = input_img[np.newaxis]
        from_axes = 'b' + from_axes
    if to_axes_c >= 0 and from_axes_c == -1:
        input_img = input_img[np.newaxis]
        from_axes = 'c' + from_axes
    return np.transpose(input_img, [from_axes.find(a) for a in to_axes])
