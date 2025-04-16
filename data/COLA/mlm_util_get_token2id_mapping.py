def get_token2id_mapping(tokenizer):
    token_list = ['before', 'after', '[none]']
    token2id_mapping = {}
    for token in token_list:
        if tokenizer.tokenize(token) != tokenizer.tokenize(' ' + token):
            token = ' ' + token
        token_id = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(token))
        assert len(token_id) <= 2
        token_id = token_id[-1]
        token2id_mapping[token.strip()] = token_id
    return token2id_mapping
