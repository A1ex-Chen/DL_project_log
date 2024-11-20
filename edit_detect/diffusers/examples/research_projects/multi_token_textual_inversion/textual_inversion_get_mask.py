def get_mask(tokenizer, accelerator):
    mask = torch.ones(len(tokenizer)).to(accelerator.device, dtype=torch.bool)
    for placeholder_token in tokenizer.token_map:
        placeholder_token_ids = tokenizer.encode(placeholder_token,
            add_special_tokens=False)
        for i in range(len(placeholder_token_ids)):
            mask = mask & (torch.arange(len(tokenizer)) !=
                placeholder_token_ids[i]).to(accelerator.device)
    return mask
