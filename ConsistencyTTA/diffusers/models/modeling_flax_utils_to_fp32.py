def to_fp32(self, params: Union[Dict, FrozenDict], mask: Any=None):
    """
        Cast the floating-point `params` to `jax.numpy.float32`. This method can be used to explicitly convert the
        model parameters to fp32 precision. This returns a new `params` tree and does not cast the `params` in place.

        Arguments:
            params (`Union[Dict, FrozenDict]`):
                A `PyTree` of model parameters.
            mask (`Union[Dict, FrozenDict]`):
                A `PyTree` with same structure as the `params` tree. The leaves should be booleans, `True` for params
                you want to cast, and should be `False` for those you want to skip

        Examples:

        ```python
        >>> from diffusers import FlaxUNet2DConditionModel

        >>> # Download model and configuration from huggingface.co
        >>> model, params = FlaxUNet2DConditionModel.from_pretrained("runwayml/stable-diffusion-v1-5")
        >>> # By default, the model params will be in fp32, to illustrate the use of this method,
        >>> # we'll first cast to fp16 and back to fp32
        >>> params = model.to_f16(params)
        >>> # now cast back to fp32
        >>> params = model.to_fp32(params)
        ```"""
    return self._cast_floating_to(params, jnp.float32, mask)
