def add_placeholder_tokens(self, placeholder_token, *args,
    num_vec_per_token=1, **kwargs):
    output = []
    if num_vec_per_token == 1:
        self.try_adding_tokens(placeholder_token, *args, **kwargs)
        output.append(placeholder_token)
    else:
        output = []
        for i in range(num_vec_per_token):
            ith_token = placeholder_token + f'_{i}'
            self.try_adding_tokens(ith_token, *args, **kwargs)
            output.append(ith_token)
    for token in self.token_map:
        if token in placeholder_token:
            raise ValueError(
                f'The tokenizer already has placeholder token {token} that can get confused with {placeholder_token}keep placeholder tokens independent'
                )
    self.token_map[placeholder_token] = output
