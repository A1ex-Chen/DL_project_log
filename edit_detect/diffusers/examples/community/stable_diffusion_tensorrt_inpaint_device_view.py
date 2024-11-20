def device_view(t):
    return cuda.DeviceView(ptr=t.data_ptr(), shape=t.shape, dtype=
        torch_to_numpy_dtype_dict[t.dtype])
