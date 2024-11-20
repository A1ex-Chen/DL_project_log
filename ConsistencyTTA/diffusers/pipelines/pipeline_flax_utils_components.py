@property
def components(self) ->Dict[str, Any]:
    """

        The `self.components` property can be useful to run different pipelines with the same weights and
        configurations to not have to re-allocate memory.

        Examples:

        ```py
        >>> from diffusers import (
        ...     FlaxStableDiffusionPipeline,
        ...     FlaxStableDiffusionImg2ImgPipeline,
        ... )

        >>> text2img = FlaxStableDiffusionPipeline.from_pretrained(
        ...     "runwayml/stable-diffusion-v1-5", revision="bf16", dtype=jnp.bfloat16
        ... )
        >>> img2img = FlaxStableDiffusionImg2ImgPipeline(**text2img.components)
        ```

        Returns:
            A dictionary containing all the modules needed to initialize the pipeline.
        """
    expected_modules, optional_parameters = self._get_signature_keys(self)
    components = {k: getattr(self, k) for k in self.config.keys() if not k.
        startswith('_') and k not in optional_parameters}
    if set(components.keys()) != expected_modules:
        raise ValueError(
            f'{self} has been incorrectly initialized or {self.__class__} is incorrectly implemented. Expected {expected_modules} to be defined, but {components} are defined.'
            )
    return components
