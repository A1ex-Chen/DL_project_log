def fn_recursive_set_mem_eff(module: torch.nn.Module):
    if hasattr(module, 'set_use_memory_efficient_attention_xformers'):
        module.set_use_memory_efficient_attention_xformers(valid, attention_op)
    for child in module.children():
        fn_recursive_set_mem_eff(child)
