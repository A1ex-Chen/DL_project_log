def set_use_memory_efficient_attention_xformers(self,
    use_memory_efficient_attention_xformers: bool, attention_op: Optional[
    Callable]=None):
    if use_memory_efficient_attention_xformers:
        if not is_xformers_available():
            raise ModuleNotFoundError(
                'Refer to https://github.com/facebookresearch/xformers for more information on how to install xformers'
                , name='xformers')
        elif not torch.cuda.is_available():
            raise ValueError(
                "torch.cuda.is_available() should be True but is False. xformers' memory efficient attention is only available for GPU "
                )
        else:
            try:
                _ = xformers.ops.memory_efficient_attention(torch.randn((1,
                    2, 40), device='cuda'), torch.randn((1, 2, 40), device=
                    'cuda'), torch.randn((1, 2, 40), device='cuda'))
            except Exception as e:
                raise e
    self._use_memory_efficient_attention_xformers = (
        use_memory_efficient_attention_xformers)
    self._attention_op = attention_op
