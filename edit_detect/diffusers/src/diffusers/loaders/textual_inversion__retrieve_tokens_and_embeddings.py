@staticmethod
def _retrieve_tokens_and_embeddings(tokens, state_dicts, tokenizer):
    all_tokens = []
    all_embeddings = []
    for state_dict, token in zip(state_dicts, tokens):
        if isinstance(state_dict, torch.Tensor):
            if token is None:
                raise ValueError(
                    'You are trying to load a textual inversion embedding that has been saved as a PyTorch tensor. Make sure to pass the name of the corresponding token in this case: `token=...`.'
                    )
            loaded_token = token
            embedding = state_dict
        elif len(state_dict) == 1:
            loaded_token, embedding = next(iter(state_dict.items()))
        elif 'string_to_param' in state_dict:
            loaded_token = state_dict['name']
            embedding = state_dict['string_to_param']['*']
        else:
            raise ValueError(
                f"""Loaded state dictionary is incorrect: {state_dict}. 

Please verify that the loaded state dictionary of the textual embedding either only has a single key or includes the `string_to_param` input key."""
                )
        if token is not None and loaded_token != token:
            logger.info(
                f'The loaded token: {loaded_token} is overwritten by the passed token {token}.'
                )
        else:
            token = loaded_token
        if token in tokenizer.get_vocab():
            raise ValueError(
                f'Token {token} already in tokenizer vocabulary. Please choose a different token name or remove {token} and embedding from the tokenizer and text encoder.'
                )
        all_tokens.append(token)
        all_embeddings.append(embedding)
    return all_tokens, all_embeddings
