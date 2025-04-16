def set_use_memory_efficient_attention_xformers(self, valid: bool,
    attention_op: Optional[Callable]=None) ->None:

    def fn_recursive_set_mem_eff(module: torch.nn.Module):
        if hasattr(module, 'set_use_memory_efficient_attention_xformers'):
            module.set_use_memory_efficient_attention_xformers(valid,
                attention_op)
        for child in module.children():
            fn_recursive_set_mem_eff(child)
    module_names, _, _ = self.extract_init_dict(dict(self.config))
    for module_name in module_names:
        module = getattr(self, module_name)
        if isinstance(module, torch.nn.Module):
            fn_recursive_set_mem_eff(module)
