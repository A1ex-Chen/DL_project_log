def set_attention_slice(self, slice_size: Optional[int]):
    module_names, _, _ = self.extract_init_dict(dict(self.config))
    for module_name in module_names:
        module = getattr(self, module_name)
        if isinstance(module, torch.nn.Module) and hasattr(module,
            'set_attention_slice'):
            module.set_attention_slice(slice_size)
