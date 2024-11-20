def tk_conv_colon_two_eval(self, conv, tokenizer, **kwargs):
    assert len(conv.messages) >= 2
    target = conv.get_prompt()
    conv.messages[-1][-1] = ''
    conversation = conv.get_prompt()
    input_ids = tokenizer([conversation], **kwargs).input_ids[0]
    target = tokenizer([target], add_special_tokens=False, **kwargs).input_ids[
        0]
    target[target == tokenizer.pad_token_id] = IGNORE_INDEX
    return dict(input_ids=input_ids, attention_mask=input_ids.ne(tokenizer.
        pad_token_id), labels=target)
