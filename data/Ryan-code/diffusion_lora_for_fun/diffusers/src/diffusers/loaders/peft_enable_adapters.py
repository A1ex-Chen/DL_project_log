def enable_adapters(self) ->None:
    """
        Enable adapters that are attached to the model. The model uses `self.active_adapters()` to retrieve the list of
        adapters to enable.

        If you are not familiar with adapters and PEFT methods, we invite you to read more about them on the PEFT
        [documentation](https://huggingface.co/docs/peft).
        """
    check_peft_version(min_version=MIN_PEFT_VERSION)
    if not self._hf_peft_config_loaded:
        raise ValueError('No adapter loaded. Please load an adapter first.')
    from peft.tuners.tuners_utils import BaseTunerLayer
    for _, module in self.named_modules():
        if isinstance(module, BaseTunerLayer):
            if hasattr(module, 'enable_adapters'):
                module.enable_adapters(enabled=True)
            else:
                module.disable_adapters = False
