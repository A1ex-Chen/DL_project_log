def __call__(self, texts, context_length=77):
    if isinstance(texts, str):
        texts = [texts]
    sot_token = self.encoder['<|startoftext|>']
    eot_token = self.encoder['<|endoftext|>']
    all_tokens = [([sot_token] + self.encode(text) + [eot_token]) for text in
        texts]
    result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)
    for i, tokens in enumerate(all_tokens):
        tokens = tokens[:context_length]
        result[i, :len(tokens)] = torch.tensor(tokens)
    if len(result) == 1:
        return result[0]
    return result
