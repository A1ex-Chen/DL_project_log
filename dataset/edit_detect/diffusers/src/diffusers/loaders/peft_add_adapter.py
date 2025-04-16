def add_adapter(self, adapter_config, adapter_name: str='default') ->None:
    """
        Adds a new adapter to the current model for training. If no adapter name is passed, a default name is assigned
        to the adapter to follow the convention of the PEFT library.

        If you are not familiar with adapters and PEFT methods, we invite you to read more about them in the PEFT
        [documentation](https://huggingface.co/docs/peft).

        Args:
            adapter_config (`[~peft.PeftConfig]`):
                The configuration of the adapter to add; supported adapters are non-prefix tuning and adaption prompt
                methods.
            adapter_name (`str`, *optional*, defaults to `"default"`):
                The name of the adapter to add. If no name is passed, a default name is assigned to the adapter.
        """
    check_peft_version(min_version=MIN_PEFT_VERSION)
    if not is_peft_available():
        raise ImportError(
            'PEFT is not available. Please install PEFT to use this function: `pip install peft`.'
            )
    from peft import PeftConfig, inject_adapter_in_model
    if not self._hf_peft_config_loaded:
        self._hf_peft_config_loaded = True
    elif adapter_name in self.peft_config:
        raise ValueError(
            f'Adapter with name {adapter_name} already exists. Please use a different name.'
            )
    if not isinstance(adapter_config, PeftConfig):
        raise ValueError(
            f'adapter_config should be an instance of PeftConfig. Got {type(adapter_config)} instead.'
            )
    adapter_config.base_model_name_or_path = None
    inject_adapter_in_model(adapter_config, self, adapter_name)
    self.set_adapter(adapter_name)
