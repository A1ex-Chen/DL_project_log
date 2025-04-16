def is_tensor(x):
    """ Tests if ``x`` is a :obj:`torch.Tensor`, :obj:`tf.Tensor` or :obj:`np.ndarray`. """
    if is_torch_available():
        import torch
        if isinstance(x, torch.Tensor):
            return True
    if is_tf_available():
        import tensorflow as tf
        if isinstance(x, tf.Tensor):
            return True
    return isinstance(x, np.ndarray)
