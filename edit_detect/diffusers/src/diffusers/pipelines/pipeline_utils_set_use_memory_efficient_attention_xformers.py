def set_use_memory_efficient_attention_xformers(self, valid: bool,
    attention_op: Optional[Callable]=None) ->None:

    def fn_recursive_set_mem_eff(module: torch.nn.Module):
        if hasattr(module, 'set_use_memory_efficient_attention_xformers'):
            module.set_use_memory_efficient_attention_xformers(valid,
                attention_op)
        for child in module.children():
            fn_recursive_set_mem_eff(child)
    module_names, _ = self._get_signature_keys(self)
    modules = [getattr(self, n, None) for n in module_names]
    modules = [m for m in modules if isinstance(m, torch.nn.Module)]
    for module in modules:
        fn_recursive_set_mem_eff(module)
