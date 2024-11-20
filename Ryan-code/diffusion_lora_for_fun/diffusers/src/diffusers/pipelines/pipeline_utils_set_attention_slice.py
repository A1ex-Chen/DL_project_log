def set_attention_slice(self, slice_size: Optional[int]):
    module_names, _ = self._get_signature_keys(self)
    modules = [getattr(self, n, None) for n in module_names]
    modules = [m for m in modules if isinstance(m, torch.nn.Module) and
        hasattr(m, 'set_attention_slice')]
    for module in modules:
        module.set_attention_slice(slice_size)
