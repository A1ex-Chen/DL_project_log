def batch_tokenizer_image_token(prompts: list, tokenizer,
    add_special_tokens=True):
    input_ids_list = []
    for prompt in prompts:
        input_ids = tokenizer_image_token(prompt, tokenizer,
            IMAGE_TOKEN_INDEX, return_tensors='pt', add_special_tokens=
            add_special_tokens)
        input_ids_list.append(input_ids)
    input_ids_list = torch.nn.utils.rnn.pad_sequence(input_ids_list,
        batch_first=True, padding_value=tokenizer.pad_token_id if
        add_special_tokens else IGNORE_INDEX)
    return input_ids_list
