def convert_tokens_to_ids(self, tokens):
    """ Converts a single token, or a sequence of tokens, (str/unicode) in a single integer id
            (resp. a sequence of ids), using the vocabulary.
        """
    if tokens is None:
        return None
    if isinstance(tokens, str) or six.PY2 and isinstance(tokens, unicode):
        return self._convert_token_to_id_with_added_voc(tokens)
    ids = []
    for token in tokens:
        ids.append(self._convert_token_to_id_with_added_voc(token))
    return ids
