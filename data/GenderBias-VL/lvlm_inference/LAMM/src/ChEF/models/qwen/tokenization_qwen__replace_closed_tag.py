def _replace_closed_tag(input_tokens: List[Any], start_tags: Union[Any,
    Tuple[Any]], end_tags: Union[Any, Tuple[Any]], inclusive_replace_func:
    Callable, exclusive_replace_func: Callable=lambda x: x):
    if isinstance(start_tags, (str, int)):
        start_tags = start_tags,
    if isinstance(end_tags, (str, int)):
        end_tags = end_tags,
    assert len(start_tags) == len(end_tags)
    output_tokens = []
    end = 0
    while True:
        start = _list_find(input_tokens, start_tags, end)
        if start == -1:
            break
        output_tokens.extend(exclusive_replace_func(input_tokens[end:start]))
        tag_idx = start_tags.index(input_tokens[start])
        end = _list_find(input_tokens, (end_tags[tag_idx],), start)
        if end == -1:
            raise ValueError('Unclosed image token')
        output_tokens.extend(inclusive_replace_func(input_tokens[start:end +
            1]))
        end += 1
    output_tokens.extend(exclusive_replace_func(input_tokens[end:]))
    return output_tokens
