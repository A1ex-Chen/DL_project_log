def __post_init__(self):
    if self.mlm and self.tokenizer.mask_token is None:
        raise ValueError(
            'This tokenizer does not have a mask token which is necessary for masked language modeling. You should pass `mlm=False` to train on causal language modeling instead.'
            )
