def get_token_map(self, prompt, padding='do_not_pad', verbose=False):
    """Get a list of mapping: prompt index to str (prompt in a list of token str)"""
    fg_prompt_tokens = self.tokenizer([prompt], padding=padding, max_length
        =77, return_tensors='np')
    input_ids = fg_prompt_tokens['input_ids'][0]
    token_map = []
    for ind, item in enumerate(input_ids.tolist()):
        token = self.tokenizer._convert_id_to_token(item)
        if verbose:
            logger.info(f'{ind}, {token} ({item})')
        token_map.append(token)
    return token_map
