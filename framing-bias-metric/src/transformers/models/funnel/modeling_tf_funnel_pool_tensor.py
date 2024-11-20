def pool_tensor(self, tensor, mode='mean', stride=2):
    """Apply 1D pooling to a tensor of size [B x T (x H)]."""
    if tensor is None:
        return None
    if isinstance(tensor, (tuple, list)):
        return type(tensor)(self.pool_tensor(tensor, mode=mode, stride=
            stride) for x in tensor)
    if self.separate_cls:
        suffix = tensor[:, :-1] if self.truncate_seq else tensor
        tensor = tf.concat([tensor[:, :1], suffix], axis=1)
    ndim = tensor.shape.ndims
    if ndim == 2:
        tensor = tensor[:, :, None]
    if mode == 'mean':
        tensor = tf.nn.avg_pool1d(tensor, stride, strides=stride,
            data_format='NWC', padding='SAME')
    elif mode == 'max':
        tensor = tf.nn.max_pool1d(tensor, stride, strides=stride,
            data_format='NWC', padding='SAME')
    elif mode == 'min':
        tensor = -tf.nn.max_pool1d(-tensor, stride, strides=stride,
            data_format='NWC', padding='SAME')
    else:
        raise NotImplementedError(
            "The supported modes are 'mean', 'max' and 'min'.")
    return tf.squeeze(tensor, 2) if ndim == 2 else tensor
