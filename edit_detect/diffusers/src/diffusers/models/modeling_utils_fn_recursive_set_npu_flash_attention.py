def fn_recursive_set_npu_flash_attention(module: torch.nn.Module):
    if hasattr(module, 'set_use_npu_flash_attention'):
        module.set_use_npu_flash_attention(valid)
    for child in module.children():
        fn_recursive_set_npu_flash_attention(child)
