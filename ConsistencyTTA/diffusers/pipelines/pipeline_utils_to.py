def to(self, torch_device: Optional[Union[str, torch.device]]=None,
    torch_dtype: Optional[torch.dtype]=None, silence_dtype_warnings: bool=False
    ):
    if torch_device is None and torch_dtype is None:
        return self

    def module_is_sequentially_offloaded(module):
        if not is_accelerate_available() or is_accelerate_version('<', '0.14.0'
            ):
            return False
        return hasattr(module, '_hf_hook') and not isinstance(module.
            _hf_hook, accelerate.hooks.CpuOffload)

    def module_is_offloaded(module):
        if not is_accelerate_available() or is_accelerate_version('<',
            '0.17.0.dev0'):
            return False
        return hasattr(module, '_hf_hook') and isinstance(module._hf_hook,
            accelerate.hooks.CpuOffload)
    pipeline_is_sequentially_offloaded = any(
        module_is_sequentially_offloaded(module) for _, module in self.
        components.items())
    if pipeline_is_sequentially_offloaded and torch.device(torch_device
        ).type == 'cuda':
        raise ValueError(
            "It seems like you have activated sequential model offloading by calling `enable_sequential_cpu_offload`, but are now attempting to move the pipeline to GPU. This is not compatible with offloading. Please, move your pipeline `.to('cpu')` or consider removing the move altogether if you use sequential offloading."
            )
    pipeline_is_offloaded = any(module_is_offloaded(module) for _, module in
        self.components.items())
    if pipeline_is_offloaded and torch.device(torch_device).type == 'cuda':
        logger.warning(
            f"It seems like you have activated model offloading by calling `enable_model_cpu_offload`, but are now manually moving the pipeline to GPU. It is strongly recommended against doing so as memory gains from offloading are likely to be lost. Offloading automatically takes care of moving the individual components {', '.join(self.components.keys())} to GPU when needed. To make sure offloading works as expected, you should consider moving the pipeline back to CPU: `pipeline.to('cpu')` or removing the move altogether if you use offloading."
            )
    module_names, _, _ = self.extract_init_dict(dict(self.config))
    is_offloaded = pipeline_is_offloaded or pipeline_is_sequentially_offloaded
    for name in module_names.keys():
        module = getattr(self, name)
        if isinstance(module, torch.nn.Module):
            module.to(torch_device, torch_dtype)
            if module.dtype == torch.float16 and str(torch_device) in ['cpu'
                ] and not silence_dtype_warnings and not is_offloaded:
                logger.warning(
                    'Pipelines loaded with `torch_dtype=torch.float16` cannot run with `cpu` device. It is not recommended to move them to `cpu` as running them will fail. Please make sure to use an accelerator to run the pipeline in inference, due to the lack of support for`float16` operations on this device in PyTorch. Please, remove the `torch_dtype=torch.float16` argument, or use another device for inference.'
                    )
    return self
