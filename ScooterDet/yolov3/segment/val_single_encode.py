def single_encode(x):
    rle = encode(np.asarray(x[:, :, None], order='F', dtype='uint8'))[0]
    rle['counts'] = rle['counts'].decode('utf-8')
    return rle
