def _collate_batch(examples, tokenizer):
    """Collate `examples` into a batch, using the information in `tokenizer` for padding if necessary."""
    if isinstance(examples[0], (list, tuple)):
        examples = [torch.tensor(e, dtype=torch.long) for e in examples]
    length_of_first = examples[0].size(0)
    are_tensors_same_length = all(x.size(0) == length_of_first for x in
        examples)
    if are_tensors_same_length:
        return torch.stack(examples, dim=0)
    if tokenizer._pad_token is None:
        raise ValueError(
            f'You are attempting to pad samples but the tokenizer you are using ({tokenizer.__class__.__name__}) does not have a pad token.'
            )
    max_length = max(x.size(0) for x in examples)
    result = examples[0].new_full([len(examples), max_length], tokenizer.
        pad_token_id)
    for i, example in enumerate(examples):
        if tokenizer.padding_side == 'right':
            result[i, :example.shape[0]] = example
        else:
            result[i, -example.shape[0]:] = example
    return result
