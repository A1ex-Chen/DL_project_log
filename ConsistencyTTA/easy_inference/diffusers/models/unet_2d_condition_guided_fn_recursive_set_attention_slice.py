def fn_recursive_set_attention_slice(module: torch.nn.Module, slice_size:
    List[int]):
    if hasattr(module, 'set_attention_slice'):
        module.set_attention_slice(slice_size.pop())
    for child in module.children():
        fn_recursive_set_attention_slice(child, slice_size)
