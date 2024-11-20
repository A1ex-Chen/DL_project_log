def fuse_text_encoder_lora(text_encoder, lora_scale=1.0, safe_fusing=False,
    adapter_names=None):
    merge_kwargs = {'safe_merge': safe_fusing}
    for module in text_encoder.modules():
        if isinstance(module, BaseTunerLayer):
            if lora_scale != 1.0:
                module.scale_layer(lora_scale)
            supported_merge_kwargs = list(inspect.signature(module.merge).
                parameters)
            if 'adapter_names' in supported_merge_kwargs:
                merge_kwargs['adapter_names'] = adapter_names
            elif 'adapter_names' not in supported_merge_kwargs and adapter_names is not None:
                raise ValueError(
                    'The `adapter_names` argument is not supported with your PEFT version. Please upgrade to the latest version of PEFT. `pip install -U peft`'
                    )
            module.merge(**merge_kwargs)
