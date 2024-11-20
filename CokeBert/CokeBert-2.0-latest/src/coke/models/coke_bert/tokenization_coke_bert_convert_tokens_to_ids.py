def convert_tokens_to_ids(self, tokens):
    """Converts a sequence of tokens into ids using the vocab."""
    ids = []
    for token in tokens:
        ids.append(self.vocab[token])
    if len(ids) > self.max_len:
        raise ValueError(
            'Token indices sequence length is longer than the specified maximum  sequence length for this BERT model ({} > {}). Running this sequence through BERT will result in indexing errors'
            .format(len(ids), self.max_len))
    return ids
