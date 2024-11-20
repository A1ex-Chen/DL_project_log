def set_attn_processor(self, processor: Union[AttentionProcessor, Dict[str,
    AttentionProcessor]]):
    """
        Sets the attention processor to use to compute attention.

        Parameters:
            processor (`dict` of `AttentionProcessor` or only `AttentionProcessor`):
                The instantiated processor class or a dictionary of processor classes that will be set as the processor
                for **all** `Attention` layers.

                If `processor` is a dict, the key needs to define the path to the corresponding cross attention
                processor. This is strongly recommended when setting trainable attention processors.

        """
    count = len(self.attn_processors.keys())
    if isinstance(processor, dict) and len(processor) != count:
        raise ValueError(
            f'A dict of processors was passed, but the number of processors {len(processor)} does not match the number of attention layers: {count}. Please make sure to pass {count} processor classes.'
            )

    def fn_recursive_attn_processor(name: str, module: torch.nn.Module,
        processor):
        if hasattr(module, 'set_processor'):
            if not isinstance(processor, dict):
                module.set_processor(processor)
            else:
                module.set_processor(processor.pop(f'{name}.processor'))
        for sub_name, child in module.named_children():
            fn_recursive_attn_processor(f'{name}.{sub_name}', child, processor)
    for name, module in self.named_children():
        fn_recursive_attn_processor(name, module, processor)
