def __init__(self, vocab_file, merges_file, tokenizer_file=None, unk_token=
    '<|endoftext|>', bos_token='<|endoftext|>', eos_token='<|endoftext|>',
    add_prefix_space=False, **kwargs):
    super().__init__(vocab_file, merges_file, tokenizer_file=tokenizer_file,
        unk_token=unk_token, bos_token=bos_token, eos_token=eos_token,
        add_prefix_space=add_prefix_space, **kwargs)
    pre_tok_state = json.loads(self.backend_tokenizer.pre_tokenizer.
        __getstate__())
    if pre_tok_state.get('add_prefix_space', add_prefix_space
        ) != add_prefix_space:
        pre_tok_class = getattr(pre_tokenizers, pre_tok_state.pop('type'))
        pre_tok_state['add_prefix_space'] = add_prefix_space
        self.backend_tokenizer.pre_tokenizer = pre_tok_class(**pre_tok_state)
    self.add_prefix_space = add_prefix_space
