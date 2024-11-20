def _fn(m):
    if isinstance(m, Caffe2Compatible):
        m.tensor_mode = enable
