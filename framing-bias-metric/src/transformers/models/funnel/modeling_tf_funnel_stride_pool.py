def stride_pool(self, tensor, axis):
    """
        Perform pooling by stride slicing the tensor along the given axis.
        """
    if tensor is None:
        return None
    if isinstance(axis, (list, tuple)):
        for ax in axis:
            tensor = self.stride_pool(tensor, ax)
        return tensor
    if isinstance(tensor, (tuple, list)):
        return type(tensor)(self.stride_pool(x, axis) for x in tensor)
    axis %= tensor.shape.ndims
    axis_slice = slice(None, -1, 2
        ) if self.separate_cls and self.truncate_seq else slice(None, None, 2)
    enc_slice = [slice(None)] * axis + [axis_slice]
    if self.separate_cls:
        cls_slice = [slice(None)] * axis + [slice(None, 1)]
        tensor = tf.concat([tensor[cls_slice], tensor], axis)
    return tensor[enc_slice]
