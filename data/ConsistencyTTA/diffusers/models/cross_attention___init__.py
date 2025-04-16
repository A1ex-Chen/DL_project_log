def __init__(self, *args, **kwargs):
    deprecation_message = (
        f"{self.__class__.__name__} is deprecated and will be removed in `0.18.0`. Please use `from diffusers.models.attention_processor import {''.join(self.__class__.__name__.split('Cross'))} instead."
        )
    deprecate('cross_attention', '0.18.0', deprecation_message,
        standard_warn=False)
    super().__init__(*args, **kwargs)
