def enable_forward_chunking(self, chunk_size: Optional[int]=None, dim: int=0
    ) ->None:
    """
        Sets the attention processor to use [feed forward
        chunking](https://huggingface.co/blog/reformer#2-chunked-feed-forward-layers).

        Parameters:
            chunk_size (`int`, *optional*):
                The chunk size of the feed-forward layers. If not specified, will run feed-forward layer individually
                over each tensor of dim=`dim`.
            dim (`int`, *optional*, defaults to `0`):
                The dimension over which the feed-forward computation should be chunked. Choose between dim=0 (batch)
                or dim=1 (sequence length).
        """
    if dim not in [0, 1]:
        raise ValueError(f'Make sure to set `dim` to either 0 or 1, not {dim}')
    chunk_size = chunk_size or 1

    def fn_recursive_feed_forward(module: torch.nn.Module, chunk_size: int,
        dim: int):
        if hasattr(module, 'set_chunk_feed_forward'):
            module.set_chunk_feed_forward(chunk_size=chunk_size, dim=dim)
        for child in module.children():
            fn_recursive_feed_forward(child, chunk_size, dim)
    for module in self.children():
        fn_recursive_feed_forward(module, chunk_size, dim)
