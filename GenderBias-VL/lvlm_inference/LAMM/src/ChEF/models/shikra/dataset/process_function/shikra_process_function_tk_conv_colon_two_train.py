def tk_conv_colon_two_train(self, conv, tokenizer, **kwargs):
    conversation = conv.get_prompt()
    input_ids = tokenizer([conversation], **kwargs).input_ids[0]
    target = copy.deepcopy(input_ids)
    assert conv.sep_style == SeparatorStyle.ADD_COLON_TWO
    sep = conv.sep + conv.roles[1] + ': '
    total_len = int(target.ne(tokenizer.pad_token_id).sum())
    rounds = conversation.split(conv.sep2)
    cur_len = 1
    target[:cur_len] = IGNORE_INDEX
    for i, rou in enumerate(rounds):
        if rou == '':
            break
        parts = rou.split(sep)
        if len(parts) != 2:
            break
        parts[0] += sep
        round_len = len(tokenizer(rou).input_ids)
        instruction_len = len(tokenizer(parts[0]).input_ids) - 2
        target[cur_len:cur_len + instruction_len] = IGNORE_INDEX
        cur_len += round_len
    target[cur_len:] = IGNORE_INDEX
    if cur_len < tokenizer.model_max_length:
        if cur_len != total_len:
            target[:] = IGNORE_INDEX
            warnings.warn(
                f"""WARNING: tokenization mismatch: {cur_len} vs. {total_len}. (ignored):
{conversation}"""
                )
    return dict(input_ids=input_ids, attention_mask=input_ids.ne(tokenizer.
        pad_token_id), labels=target)
