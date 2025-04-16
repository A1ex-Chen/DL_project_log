def __init__(self, min_length_for_response=32, *args, **kwargs):
    super().__init__(*args, **kwargs)
    assert self.tokenizer.eos_token_id is not None, 'DialoguePipeline tokenizer should have an EOS token set'
    if self.tokenizer.pad_token_id is None:
        self.tokenizer.pad_token = self.tokenizer.eos_token
    self.min_length_for_response = min_length_for_response
