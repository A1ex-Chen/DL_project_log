def check_sliceable_dim_attr(module: torch.nn.Module):
    if hasattr(module, 'set_attention_slice'):
        assert isinstance(module.sliceable_head_dim, int)
    for child in module.children():
        check_sliceable_dim_attr(child)
