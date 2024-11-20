def _get_tensor_dtypes(dataloader, precision):

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
    input_dtypes = {}
    output_dtypes = {}
    for batch in dataloader:
        _, x, y = batch
        input_dtypes = _get_dtypes(x)
        output_dtypes = _get_dtypes(y)
        break
    return input_dtypes, output_dtypes
