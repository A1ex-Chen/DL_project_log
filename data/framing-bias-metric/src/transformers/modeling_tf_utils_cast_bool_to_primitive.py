def cast_bool_to_primitive(bool_variable: Union[tf.Tensor, bool],
    default_tensor_to_true=False) ->bool:
    """
    Function arguments can be inserted as boolean tensor and bool variables to cope with Keras serialization we need to
    cast the bool arguments (like :obj:`output_attentions` for instance) to correct boolean if it is a tensor.

    Args:
        bool_variable (:obj:`Union[tf.Tensor, bool]`):
            The variable to convert to a boolean.
        default_tensor_to_true (:obj:`bool`, `optional`, defaults to `False`):
            The default value to use in case the tensor has no numpy attribute.

    Returns:
        :obj:`bool`: The converted value.
    """
    if tf.is_tensor(bool_variable):
        if hasattr(bool_variable, 'numpy'):
            return bool(bool_variable.numpy())
        elif default_tensor_to_true:
            return True
    return bool_variable
