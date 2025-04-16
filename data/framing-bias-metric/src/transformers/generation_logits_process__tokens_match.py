def _tokens_match(self, prev_tokens: torch.LongTensor, tokens: List[int]
    ) ->bool:
    if len(tokens) == 0:
        return True
    elif len(tokens) > len(prev_tokens):
        return False
    elif prev_tokens[-len(tokens):].tolist() == tokens:
        return True
    else:
        return False
