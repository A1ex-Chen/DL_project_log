def _fuse_lora_apply(self, module, adapter_names=None):
    if not USE_PEFT_BACKEND:
        if hasattr(module, '_fuse_lora'):
            module._fuse_lora(self.lora_scale, self._safe_fusing)
        if adapter_names is not None:
            raise ValueError(
                'The `adapter_names` argument is not supported in your environment. Please switch to PEFT backend to use this argument by installing latest PEFT and transformers. `pip install -U peft transformers`'
                )
    else:
        from peft.tuners.tuners_utils import BaseTunerLayer
        merge_kwargs = {'safe_merge': self._safe_fusing}
        if isinstance(module, BaseTunerLayer):
            if self.lora_scale != 1.0:
                module.scale_layer(self.lora_scale)
            supported_merge_kwargs = list(inspect.signature(module.merge).
                parameters)
            if 'adapter_names' in supported_merge_kwargs:
                merge_kwargs['adapter_names'] = adapter_names
            elif 'adapter_names' not in supported_merge_kwargs and adapter_names is not None:
                raise ValueError(
                    'The `adapter_names` argument is not supported with your PEFT version. Please upgrade to the latest version of PEFT. `pip install -U peft`'
                    )
            module.merge(**merge_kwargs)
