def __init__(self, layerdrop=0.0, pre_norm=False, pad_token_id=2,
    bos_token_id=0, **kwargs):
    """Constructs FlaubertConfig."""
    super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id,
        **kwargs)
    self.layerdrop = layerdrop
    self.pre_norm = pre_norm
