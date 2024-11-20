def _clear_jit_cache():
    from torch.jit._recursive import concrete_type_store
    from torch.jit._state import _jit_caching_layer
    concrete_type_store.type_store.clear()
    _jit_caching_layer.clear()
