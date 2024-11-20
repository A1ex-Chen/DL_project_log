def upsample3d_nn(x):
    xshape = x.shape
    yshape = 2 * xshape[0], 2 * xshape[1], 2 * xshape[2]
    y = np.zeros(yshape, dtype=x.dtype)
    y[::2, ::2, ::2] = x
    y[::2, ::2, 1::2] = x
    y[::2, 1::2, ::2] = x
    y[::2, 1::2, 1::2] = x
    y[1::2, ::2, ::2] = x
    y[1::2, ::2, 1::2] = x
    y[1::2, 1::2, ::2] = x
    y[1::2, 1::2, 1::2] = x
    return y
