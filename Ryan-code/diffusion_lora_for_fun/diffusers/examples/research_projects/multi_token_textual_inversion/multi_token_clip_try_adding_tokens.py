def try_adding_tokens(self, placeholder_token, *args, **kwargs):
    num_added_tokens = super().add_tokens(placeholder_token, *args, **kwargs)
    if num_added_tokens == 0:
        raise ValueError(
            f'The tokenizer already contains the token {placeholder_token}. Please pass a different `placeholder_token` that is not already in the tokenizer.'
            )
