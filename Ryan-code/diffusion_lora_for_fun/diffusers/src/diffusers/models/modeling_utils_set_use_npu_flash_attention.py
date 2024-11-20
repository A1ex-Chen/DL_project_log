def set_use_npu_flash_attention(self, valid: bool) ->None:
    """
        Set the switch for the npu flash attention.
        """

    def fn_recursive_set_npu_flash_attention(module: torch.nn.Module):
        if hasattr(module, 'set_use_npu_flash_attention'):
            module.set_use_npu_flash_attention(valid)
        for child in module.children():
            fn_recursive_set_npu_flash_attention(child)
    for module in self.children():
        if isinstance(module, torch.nn.Module):
            fn_recursive_set_npu_flash_attention(module)
