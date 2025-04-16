def active_adapters(self) ->List[str]:
    """
        Gets the current list of active adapters of the model.

        If you are not familiar with adapters and PEFT methods, we invite you to read more about them on the PEFT
        [documentation](https://huggingface.co/docs/peft).
        """
    check_peft_version(min_version=MIN_PEFT_VERSION)
    if not is_peft_available():
        raise ImportError(
            'PEFT is not available. Please install PEFT to use this function: `pip install peft`.'
            )
    if not self._hf_peft_config_loaded:
        raise ValueError('No adapter loaded. Please load an adapter first.')
    from peft.tuners.tuners_utils import BaseTunerLayer
    for _, module in self.named_modules():
        if isinstance(module, BaseTunerLayer):
            return module.active_adapter
