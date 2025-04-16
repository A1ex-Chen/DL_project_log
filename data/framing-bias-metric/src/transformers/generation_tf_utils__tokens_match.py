def _tokens_match(prev_tokens, tokens):
    if len(tokens) == 0:
        return True
    if len(tokens) > len(prev_tokens):
        return False
    if prev_tokens[-len(tokens):] == tokens:
        return True
    else:
        return False
