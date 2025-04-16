def _check_text_inv_inputs(self, tokenizer, text_encoder,
    pretrained_model_name_or_paths, tokens):
    if tokenizer is None:
        raise ValueError(
            f'{self.__class__.__name__} requires `self.tokenizer` or passing a `tokenizer` of type `PreTrainedTokenizer` for calling `{self.load_textual_inversion.__name__}`'
            )
    if text_encoder is None:
        raise ValueError(
            f'{self.__class__.__name__} requires `self.text_encoder` or passing a `text_encoder` of type `PreTrainedModel` for calling `{self.load_textual_inversion.__name__}`'
            )
    if len(pretrained_model_name_or_paths) > 1 and len(
        pretrained_model_name_or_paths) != len(tokens):
        raise ValueError(
            f'You have passed a list of models of length {len(pretrained_model_name_or_paths)}, and list of tokens of length {len(tokens)} Make sure both lists have the same length.'
            )
    valid_tokens = [t for t in tokens if t is not None]
    if len(set(valid_tokens)) < len(valid_tokens):
        raise ValueError(
            f'You have passed a list of tokens that contains duplicates: {tokens}'
            )
