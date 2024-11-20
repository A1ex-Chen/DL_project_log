def fn_recursive_add_processors(name: str, module: torch.nn.Module,
    processors: Dict[str, AttentionProcessor]):
    if hasattr(module, 'get_processor'):
        processors[f'{name}.processor'] = module.get_processor(
            return_deprecated_lora=True)
    for sub_name, child in module.named_children():
        fn_recursive_add_processors(f'{name}.{sub_name}', child, processors)
    return processors
