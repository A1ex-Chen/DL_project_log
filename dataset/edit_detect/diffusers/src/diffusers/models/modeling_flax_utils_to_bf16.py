def to_bf16(self, params: Union[Dict, FrozenDict], mask: Any=None):
    """
        Cast the floating-point `params` to `jax.numpy.bfloat16`. This returns a new `params` tree and does not cast
        the `params` in place.

        This method can be used on a TPU to explicitly convert the model parameters to bfloat16 precision to do full
        half-precision training or to save weights in bfloat16 for inference in order to save memory and improve speed.

        Arguments:
            params (`Union[Dict, FrozenDict]`):
                A `PyTree` of model parameters.
            mask (`Union[Dict, FrozenDict]`):
                A `PyTree` with same structure as the `params` tree. The leaves should be booleans. It should be `True`
                for params you want to cast, and `False` for those you want to skip.

        Examples:

        ```python
        >>> from diffusers import FlaxUNet2DConditionModel

        >>> # load model
        >>> model, params = FlaxUNet2DConditionModel.from_pretrained("runwayml/stable-diffusion-v1-5")
        >>> # By default, the model parameters will be in fp32 precision, to cast these to bfloat16 precision
        >>> params = model.to_bf16(params)
        >>> # If you don't want to cast certain parameters (for example layer norm bias and scale)
        >>> # then pass the mask as follows
        >>> from flax import traverse_util

        >>> model, params = FlaxUNet2DConditionModel.from_pretrained("runwayml/stable-diffusion-v1-5")
        >>> flat_params = traverse_util.flatten_dict(params)
        >>> mask = {
        ...     path: (path[-2] != ("LayerNorm", "bias") and path[-2:] != ("LayerNorm", "scale"))
        ...     for path in flat_params
        ... }
        >>> mask = traverse_util.unflatten_dict(mask)
        >>> params = model.to_bf16(params, mask)
        ```"""
    return self._cast_floating_to(params, jnp.bfloat16, mask)
