def set_adapter(self, adapter_name: Union[str, List[str]]) ->None:
    """
        Sets a specific adapter by forcing the model to only use that adapter and disables the other adapters.

        If you are not familiar with adapters and PEFT methods, we invite you to read more about them on the PEFT
        [documentation](https://huggingface.co/docs/peft).

        Args:
            adapter_name (Union[str, List[str]])):
                The list of adapters to set or the adapter name in the case of a single adapter.
        """
    check_peft_version(min_version=MIN_PEFT_VERSION)
    if not self._hf_peft_config_loaded:
        raise ValueError('No adapter loaded. Please load an adapter first.')
    if isinstance(adapter_name, str):
        adapter_name = [adapter_name]
    missing = set(adapter_name) - set(self.peft_config)
    if len(missing) > 0:
        raise ValueError(
            f"Following adapter(s) could not be found: {', '.join(missing)}. Make sure you are passing the correct adapter name(s). current loaded adapters are: {list(self.peft_config.keys())}"
            )
    from peft.tuners.tuners_utils import BaseTunerLayer
    _adapters_has_been_set = False
    for _, module in self.named_modules():
        if isinstance(module, BaseTunerLayer):
            if hasattr(module, 'set_adapter'):
                module.set_adapter(adapter_name)
            elif not hasattr(module, 'set_adapter') and len(adapter_name) != 1:
                raise ValueError(
                    'You are trying to set multiple adapters and you have a PEFT version that does not support multi-adapter inference. Please upgrade to the latest version of PEFT. `pip install -U peft` or `pip install -U git+https://github.com/huggingface/peft.git`'
                    )
            else:
                module.active_adapter = adapter_name
            _adapters_has_been_set = True
    if not _adapters_has_been_set:
        raise ValueError(
            'Did not succeeded in setting the adapter. Please make sure you are using a model that supports adapters.'
            )
