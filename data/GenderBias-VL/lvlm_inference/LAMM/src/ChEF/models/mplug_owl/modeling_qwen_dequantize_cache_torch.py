def dequantize_cache_torch(qdata, scale, zero):
    data = scale * (qdata - zero)
    return data
