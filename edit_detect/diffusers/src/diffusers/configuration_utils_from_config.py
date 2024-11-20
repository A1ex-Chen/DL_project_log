@classmethod
def from_config(cls, config: Union[FrozenDict, Dict[str, Any]]=None,
    return_unused_kwargs=False, **kwargs):
    """
        Instantiate a Python class from a config dictionary.

        Parameters:
            config (`Dict[str, Any]`):
                A config dictionary from which the Python class is instantiated. Make sure to only load configuration
                files of compatible classes.
            return_unused_kwargs (`bool`, *optional*, defaults to `False`):
                Whether kwargs that are not consumed by the Python class should be returned or not.
            kwargs (remaining dictionary of keyword arguments, *optional*):
                Can be used to update the configuration object (after it is loaded) and initiate the Python class.
                `**kwargs` are passed directly to the underlying scheduler/model's `__init__` method and eventually
                overwrite the same named arguments in `config`.

        Returns:
            [`ModelMixin`] or [`SchedulerMixin`]:
                A model or scheduler object instantiated from a config dictionary.

        Examples:

        ```python
        >>> from diffusers import DDPMScheduler, DDIMScheduler, PNDMScheduler

        >>> # Download scheduler from huggingface.co and cache.
        >>> scheduler = DDPMScheduler.from_pretrained("google/ddpm-cifar10-32")

        >>> # Instantiate DDIM scheduler class with same config as DDPM
        >>> scheduler = DDIMScheduler.from_config(scheduler.config)

        >>> # Instantiate PNDM scheduler class with same config as DDPM
        >>> scheduler = PNDMScheduler.from_config(scheduler.config)
        ```
        """
    if 'pretrained_model_name_or_path' in kwargs:
        config = kwargs.pop('pretrained_model_name_or_path')
    if config is None:
        raise ValueError(
            'Please make sure to provide a config as the first positional argument.'
            )
    if not isinstance(config, dict):
        deprecation_message = (
            'It is deprecated to pass a pretrained model name or path to `from_config`.'
            )
        if 'Scheduler' in cls.__name__:
            deprecation_message += (
                f'If you were trying to load a scheduler, please use {cls}.from_pretrained(...) instead. Otherwise, please make sure to pass a configuration dictionary instead. This functionality will be removed in v1.0.0.'
                )
        elif 'Model' in cls.__name__:
            deprecation_message += (
                f'If you were trying to load a model, please use {cls}.load_config(...) followed by {cls}.from_config(...) instead. Otherwise, please make sure to pass a configuration dictionary instead. This functionality will be removed in v1.0.0.'
                )
        deprecate('config-passed-as-path', '1.0.0', deprecation_message,
            standard_warn=False)
        config, kwargs = cls.load_config(pretrained_model_name_or_path=
            config, return_unused_kwargs=True, **kwargs)
    init_dict, unused_kwargs, hidden_dict = cls.extract_init_dict(config,
        **kwargs)
    if 'dtype' in unused_kwargs:
        init_dict['dtype'] = unused_kwargs.pop('dtype')
    for deprecated_kwarg in cls._deprecated_kwargs:
        if deprecated_kwarg in unused_kwargs:
            init_dict[deprecated_kwarg] = unused_kwargs.pop(deprecated_kwarg)
    model = cls(**init_dict)
    if '_class_name' in hidden_dict:
        hidden_dict['_class_name'] = cls.__name__
    model.register_to_config(**hidden_dict)
    unused_kwargs = {**unused_kwargs, **hidden_dict}
    if return_unused_kwargs:
        return model, unused_kwargs
    else:
        return model
