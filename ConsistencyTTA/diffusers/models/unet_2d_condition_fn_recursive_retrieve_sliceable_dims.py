def fn_recursive_retrieve_sliceable_dims(module: torch.nn.Module):
    if hasattr(module, 'set_attention_slice'):
        sliceable_head_dims.append(module.sliceable_head_dim)
    for child in module.children():
        fn_recursive_retrieve_sliceable_dims(child)
