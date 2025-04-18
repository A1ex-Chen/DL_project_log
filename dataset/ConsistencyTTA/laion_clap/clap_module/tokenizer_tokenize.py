def tokenize(texts: Union[str, List[str]], context_length: int=77
    ) ->torch.LongTensor:
    """
    Returns the tokenized representation of given input string(s)

    Parameters
    ----------
    texts : Union[str, List[str]]
        An input string or a list of input strings to tokenize
    context_length : int
        The context length to use; all CLIP models use 77 as the context length

    Returns
    -------
    A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length]
    """
    if isinstance(texts, str):
        texts = [texts]
    sot_token = _tokenizer.encoder['<start_of_text>']
    eot_token = _tokenizer.encoder['<end_of_text>']
    all_tokens = [([sot_token] + _tokenizer.encode(text) + [eot_token]) for
        text in texts]
    result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)
    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            tokens = tokens[:context_length]
        result[i, :len(tokens)] = torch.tensor(tokens)
    return result
