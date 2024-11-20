def shape_list(tensor: tf.Tensor) ->List[int]:
    """
    Deal with dynamic shape in tensorflow cleanly.

    Args:
        tensor (:obj:`tf.Tensor`): The tensor we want the shape of.

    Returns:
        :obj:`List[int]`: The shape of the tensor as a list.
    """
    dynamic = tf.shape(tensor)
    if tensor.shape == tf.TensorShape(None):
        return dynamic.as_list()
    static = tensor.shape.as_list()
    return [(dynamic[i] if s is None else s) for i, s in enumerate(static)]
