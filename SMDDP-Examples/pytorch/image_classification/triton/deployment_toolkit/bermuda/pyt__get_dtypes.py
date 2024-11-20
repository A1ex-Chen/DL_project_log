def _get_dtypes(t):
    dtypes = {}
    for k, v in t.items():
        dtype = str(v.dtype)
        if dtype == 'float64':
            dtype = 'float32'
        if precision == Precision.FP16 and dtype == 'float32':
            dtype = 'float16'
        dtypes[k] = dtype
    return dtypes
