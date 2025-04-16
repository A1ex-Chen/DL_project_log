def batch_tokenizer_image_token(prompts: list, tokenizer,
    add_special_tokens=True):
    input_ids_list = []
    for prompt in prompts:
        input_ids = tokenizer_image_token(prompt, tokenizer,
            IMAGE_TOKEN_INDEX, return_tensors='pt').tolist()
        input_ids_list.append(input_ids)
    input_tokens_max_length = max([len(x) for x in input_ids_list])
    pad_token_id = (tokenizer.pad_token_id if add_special_tokens else
        IGNORE_INDEX)
    input_ids_list = [([pad_token_id] * (input_tokens_max_length - len(_)) +
        _) for _ in input_ids_list]
    input_ids_list = torch.LongTensor(input_ids_list)
    if add_special_tokens:
        attention_mask = 1 - input_ids_list.eq(pad_token_id).long()
        return input_ids_list, attention_mask
    else:
        return input_ids_list
