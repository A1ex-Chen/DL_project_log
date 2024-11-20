def _extract_token_prob(self, arr, token, crop=1):
    for it in arr:
        if len(it['token_str']) >= crop and token == it['token_str'][crop:]:
            return it['score']
    return 0.0
