def convert_to_tensors(self, tensor_type: Optional[Union[str, TensorType]]=
    None, prepend_batch_axis: bool=False):
    """
        Convert the inner content to tensors.

        Args:
            tensor_type (:obj:`str` or :class:`~transformers.tokenization_utils_base.TensorType`, `optional`):
                The type of tensors to use. If :obj:`str`, should be one of the values of the enum
                :class:`~transformers.tokenization_utils_base.TensorType`. If :obj:`None`, no modification is done.
            prepend_batch_axis (:obj:`int`, `optional`, defaults to :obj:`False`):
                Whether or not to add the batch dimension during the conversion.
        """
    if tensor_type is None:
        return self
    if not isinstance(tensor_type, TensorType):
        tensor_type = TensorType(tensor_type)
    if tensor_type == TensorType.TENSORFLOW:
        if not is_tf_available():
            raise ImportError(
                'Unable to convert output to TensorFlow tensors format, TensorFlow is not installed.'
                )
        as_tensor = tf.constant
        is_tensor = tf.is_tensor
    elif tensor_type == TensorType.PYTORCH:
        if not is_torch_available():
            raise ImportError(
                'Unable to convert output to PyTorch tensors format, PyTorch is not installed.'
                )
        as_tensor = torch.tensor
        is_tensor = torch.is_tensor
    elif tensor_type == TensorType.JAX:
        if not is_flax_available():
            raise ImportError(
                'Unable to convert output to JAX tensors format, JAX is not installed.'
                )
        as_tensor = jnp.array
        is_tensor = _is_jax
    else:
        as_tensor = np.asarray
        is_tensor = _is_numpy
    for key, value in self.items():
        try:
            if prepend_batch_axis:
                value = [value]
            if not is_tensor(value):
                tensor = as_tensor(value)
                self[key] = tensor
        except:
            if key == 'overflowing_tokens':
                raise ValueError(
                    'Unable to create tensor returning overflowing tokens of different lengths. Please see if a fast version of this tokenizer is available to have this feature available.'
                    )
            raise ValueError(
                "Unable to create tensor, you should probably activate truncation and/or padding with 'padding=True' 'truncation=True' to have batched tensors with the same length."
                )
    return self
