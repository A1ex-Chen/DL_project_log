def disable_forward_chunking(self):

    def fn_recursive_feed_forward(module: torch.nn.Module, chunk_size: int,
        dim: int):
        if hasattr(module, 'set_chunk_feed_forward'):
            module.set_chunk_feed_forward(chunk_size=chunk_size, dim=dim)
        for child in module.children():
            fn_recursive_feed_forward(child, chunk_size, dim)
    for module in self.children():
        fn_recursive_feed_forward(module, None, 0)
