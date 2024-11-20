def convert_mask(m, shape):
    if isinstance(m, BitMasks):
        return mm_BitMasks(m.tensor.cpu().numpy(), shape[0], shape[1])
    else:
        return mm_PolygonMasks(m.polygons, shape[0], shape[1])
