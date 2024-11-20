def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
    if hasattr(module, 'set_processor'):
        if not isinstance(processor, dict):
            module.set_processor(processor)
        else:
            module.set_processor(processor.pop(f'{name}.processor'))
    for sub_name, child in module.named_children():
        fn_recursive_attn_processor(f'{name}.{sub_name}', child, processor)
