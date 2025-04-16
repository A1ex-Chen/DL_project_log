def set_caffe2_compatible_tensor_mode(model, enable=True):

    def _fn(m):
        if isinstance(m, Caffe2Compatible):
            m.tensor_mode = enable
    model.apply(_fn)
